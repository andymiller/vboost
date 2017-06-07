"""
Black box variational inference using a multivariate normal approximation
with a low-rank + off diagonal covariance.  The variational distribution
is

    q(x; lam) = N(mu(lam), Sigma(lam))

    where

    Sigma(lam) = C(lam) C(lam)^T + diag(s(lam))

Computing the entropy for a low-rank + diagonal MVN is cubic in the low rank
part.
"""
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from . import lowr_mvn, misc, components, optimizers


default_options = {
   'num_samples'      : 100,
   'gradient_type'    : "standard",
   'fix_samples'      : False,
   'num_iters'        : 200,
   'step_size'        : .5,
   'forgetting_rate'  : .75,
   'init_params'      : None,
   'init_mean'        : None,
   'init_lowr'        : None,
   'init_stds'        : None,
   'beta_data'        : 1.,
   'beta_entropy'     : 1.,
   'full_monte_carlo' : False,
   'track_params'     : False,
   'clamped_m'        : None,
   'clamped_v'        : None,
   'clamped_C'        : None,
   'print_gap'        : 100,
   'val_fun'          : None,
   'min_iter'         : 10
}


class LowRankVI:
    """
    Class for Reparam Variational Inference with Spiked Covariance Components

    Main entry point: this returns a diagonal normal and an ELBO trace
    given an unnormalized PDF, lnpdf

    Inputs
    ------
    lnpdf : unnormalized distribution, lnpdf(th, i)
    D     : dimensionality of the RV for lnpdf
    rank  : the number of ranks used, (between 0 and D)

    Optional
    --------
    num_samples      : number of draws for the Monte Carlo estimate
    num_iters        : number of iterations to run
    ...

    """
    def __init__(self, lnpdf, D, rank, **kwargs):
        self.lnpdf = lnpdf
        self.D     = D
        self.rank  = rank

        # handle all default options
        self.args = default_options.copy()
        for k, v in kwargs.iteritems():
            self.args[k] = v

        # create the mc elbo function, gradient and component class
        self.objective, gradient, self.comp = \
                low_rank_mvn_bbvi(lnpdf, D, rank,
                                  clamped_m = self.args['clamped_m'],
                                  clamped_v = self.args['clamped_v'],
                                  clamped_C = self.args['clamped_C'])
        self.gradient = self._make_gradient(gradient)

        # create step size schedule for optimizaiton
        ni, ss, fr = self.args['num_iters'], \
                     self.args['step_size'], \
                     self.args['forgetting_rate']
        self.step_size_schedule = np.power(np.arange(ni) + 1./ss, -fr)

        # create callback for optimization routine
        self.elbo_vals = []

        # if we have a validation function, create a break condition function
        self.val_error = np.inf

    def fit(self):
        # initialize parameter values
        init_params = self._init_params()
        variational_params = \
            optimizers.adam(self.gradient, init_params,
                            step_size  = self.step_size_schedule,
                            num_iters  = self.args['num_iters'],
                            callback   = self._callback,
                            break_cond = self._break_cond)
        self.comp.lam = variational_params
        return self.comp

    def _break_cond(self, x, i, g):
        if self.args['val_fun'] is None:
            return False

        if (i % 20 == 0) and i > self.args['min_iter']:
            th_samps = self.comp.sample(400, lam=x)
            verror = self.args['val_fun'](th_samps)
            if verror > self.val_error:
                print "   ... breaking with validation error: ", verror
                return True
            self.val_error = verror
            return False

    def _init_params(self):
        # initialize variational parameters
        print "Initializing new component: ", self.comp
        init_params = np.zeros(self.comp.n_params)

        # default inits
        init_mean    = .001*np.random.randn(self.D)
        init_log_std = -2*np.ones(self.D)
        init_lowr    = np.zeros((self.D, self.comp.free_ranks))
        if self.args['init_mean'] is not None:
            init_mean = self.args['init_mean']

        if self.args['init_stds'] is not None:
            init_log_std = np.log(self.args['init_stds'])

        if self.args['init_lowr'] is not None:
            assert init_lowr.shape == (D, self.comp.free_ranks)
            init_lowr = self.args['init_lowr']

        # this stuff here can be in the component...
        if self.args['clamped_m'] is None:
            init_params = self.comp.setter(init_params, mean = init_mean)
        if self.args['clamped_v'] is None:
            init_params = self.comp.setter(init_params, v = 2*init_log_std)
        init_params = self.comp.setter(init_params, C=init_lowr)
        return init_params

    def _callback(self, x, i, g):
        elbo_val = -self.objective(x, i, eps_lowr=self.eps_lowr,
                                         eps_diag=self.eps_diag,
                                         full_monte_carlo=True)
        if self.args['track_params']:
            self.elbo_vals.append((elbo_val, x))
        else:
            self.elbo_vals.append(elbo_val)

        if i % self.args['print_gap'] == 0:
            print "iter %d, lower bound %2.4f (step_size %2.4f)" % \
                (i, elbo_val, self.step_size_schedule[i])
            elbo_val = -self.objective(x, i,
                                       eps_lowr=self.eps_lowr,
                                       eps_diag=self.eps_diag,
                                       print_terms=True)

            if self.args['val_fun'] is not None:
                th_samps = self.comp.sample(100, lam=x)
                verror = self.args['val_fun'](th_samps)
                print "    val error: %2.4f"%verror

    def _make_gradient(self, gradient):
        # fix epsilon to 100 samples
        eps_lowr, eps_diag = None, None
        if self.args['fix_samples']:
            eps_lowr = npr.randn(num_samples, self.rank)
            eps_diag = npr.randn(num_samples, self.D)
        self.eps_lowr = eps_lowr
        self.eps_diag = eps_diag

        gradient_type = self.args['gradient_type']
        gfun = lambda x, t: gradient(x, t,
                             num_samples=self.args['num_samples'],
                             eps_lowr = eps_lowr,
                             eps_diag = eps_diag,
                             beta_data = self.args['beta_data'],
                             beta_entropy = self.args['beta_entropy'],
                             full_monte_carlo = self.args['full_monte_carlo'])

        # create objective/gradient
        if gradient_type == "natural":
            #raise NotImplementedError("Nat Grad unimplemented")
            def gradient_fixed(x, t):
                mean, C, v = unpack_params(x)
                finv = misc.mvn_fisher_info(np.concatenate([mean, .5*v]))
                return (1./finv)*gfun(x, t)

        elif gradient_type == "standard":
            gradient_fixed = gfun

        elif gradient_type == "remove_score":
            raise NotImplementedError("Only natural and standard gradients here")
            gradient_fixed = gfun
            # compute control variate for each sample
            def qlnpdf(lam, samps):
                mean, C, v = unpack_params(lam)
                return lowr_mvn.mvn_lowrank_logpdf(samps, mean, C, v)
        else:
            raise NotImplementedError("Only natural and standard gradients here")

        return gradient_fixed


##################################################
## Mean Field Variational Inference Function 
##################################################
def fit_mfvi(lnpdf, D, rank, **kwargs):
    lrvi = LowRankVI(lnpdf, D, rank, **kwargs)
    return lrvi.fit(), lrvi


##############################
# Optimization objective     #
##############################

def low_rank_mvn_bbvi(logprob, D, rank, clamped_m=None,
                                        clamped_v=None,
                                        clamped_C=None):
    """ Makes functions for low rank mvn variational inference
    """
    # organize variational parameters --- mean and low rank covariance
    comp = components.LRDComponent(D, rank, clamped_m=clamped_m,
                                            clamped_v=clamped_v,
                                            clamped_C=clamped_C)

    # define the elbo
    rs = npr.RandomState(0)
    def mc_elbo(params, t, num_samples=10,
                        eps_lowr=None, eps_diag=None,
                        beta_entropy=1., beta_data=1.,
                        full_monte_carlo=False,
                        print_terms=False):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, C, v = comp.unpack(params)
        samples    = lowr_mvn.mvn_lowrank_sample(num_samples, mean, C, v,
                                                 eps_lowr=eps_lowr,
                                                 eps_diag=eps_diag)

        # estimate evidence lower bound
        if full_monte_carlo:
            """ don't use closed form entropy """
            data_term   = beta_data * logprob(samples, t)
            ent_term    = -beta_entropy * \
                          lowr_mvn.mvn_lowrank_logpdf(samples, mean, C, v)
            lower_bound = np.mean(data_term + ent_term)
        else:
            """ use closed form entropy """
            data_term   = beta_data    * np.mean(logprob(samples, t))
            ent_term    = beta_entropy * lowr_mvn.mvn_lowrank_entropy(C, v)
            lower_bound = np.mean(data_term) + ent_term

        if print_terms:
            print "data, entropy: %2.4f, %2.4f "%(np.mean(data_term), np.mean(ent_term))

        return -lower_bound

    gradient = grad(mc_elbo)
    return mc_elbo, gradient, comp

