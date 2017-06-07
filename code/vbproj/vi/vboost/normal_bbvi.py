""" Deprecated --- use mvn_bbvi.py """
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

#################################################
# Mean Field Variational Inference Function 
#################################################
def fit_mfvi(lnpdf, D, num_samples=100,
                       gradient_type = "natural",
                       fix_samples   = False,
                       num_iters     = 200,
                       step_size     = .5, 
                       init_params   = None,
                       init_mean     = None,
                       init_stds     = None,
                       beta_data     = 1.,
                       beta_entropy  = 1.):
    """ Main entry point: this returns a diagonal normal and an ELBO trace
        given an unnormalized PDF, lnpdf
    """
    objective, gradient, unpack_params = \
        black_box_variational_inference(lnpdf, D, num_samples=num_samples)

    # fix epsilon to 100 samples
    eps = npr.randn(num_samples, D) if fix_samples else None
    if gradient_type == "natural":
        gradient_fixed = \
            lambda x, t: (1./fisher_info(x)) * \
                         gradient(x, t, eps=eps, beta_data=beta_data,
                                                 beta_entropy=beta_entropy)
    elif gradient_type == "standard":
        gradient_fixed = \
            lambda x, t: gradient(x, t, eps=eps, beta_data=beta_data,
                                                 beta_entropy=beta_entropy)
    else:
        raise NotImplementedError("Only natural and standard gradients here")

    # first fit a single gaussian using BBVI
    elbo_vals = []
    def callback(x, i, g):
        elbo_val = -objective(x, i, eps=eps)
        elbo_vals.append(elbo_val)
        if i % 10 == 0:
            print "iter %d, lower bound %2.4f"%(i, elbo_val)

    print("Optimizing variational parameters...")
    from VarBoost.optimizers import adam, sgd
    if init_mean is None:
        init_mean       = .001* np.random.randn(D)
    if init_stds is None:
        init_log_std    = -2 * np.ones(D)
    else:
        init_log_std    = np.log(init_stds)

    if init_params is None:
        init_params = np.concatenate([init_mean, init_log_std])

    variational_params = sgd(gradient_fixed, init_params, step_size=step_size,
                             mass=.00001, num_iters=num_iters, callback=callback)
    return variational_params, np.array(elbo_vals)


def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t, eps=None, beta_entropy=1., beta_data=1.):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        if eps is None:
            eps = rs.randn(num_samples, D)
        samples     = eps * np.exp(log_std) + mean
        lower_bound = beta_entropy * gaussian_entropy(log_std) + \
                      beta_data * np.mean(logprob(samples, t))
        return -lower_bound

    gradient = grad(variational_objective)
    return variational_objective, gradient, unpack_params


def fisher_info(params):
    """ returns the fisher information matrix (diagonal) for a multivariate
    normal distribution with params = [mu, ln sigma] """
    D = len(params) / 2
    mean, log_std = params[:D], params[D:]
    return np.concatenate([np.exp(-2.*log_std),
                           2*np.ones(D)])

