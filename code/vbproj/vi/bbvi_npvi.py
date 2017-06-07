"""
Port of Nonparametric Variational Inference (Gershman et al)
matlab code to python
"""
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian, hessian
from scipy.stats import ncx2
from aip.misc import mvn_diag_logpdf, mvn_diag_entropy
#from .bbvi_base import BBVI

from scipy.optimize import minimize
import autograd.scipy.misc as scpm

class NPVI(object):

    def __init__(self, lnpdf, D):
        """ Nonparametric Variational Inference Object
            INPUTS:
              lnpdf - function handle for negative log joint pdf.  should
                      be an 'autograd-able' function
        """
        self.lnpdf = lnpdf
        self.D     = D

        # enforce minimium bandwidth to avoid numerical problems
        self.s2min = 1e-7

    def run(self, theta, niter=10, tol=.0001, verbose=False):
        """ runs NPV for ... iterations 
            mimics npv_run.m from Sam Gershman's original matlab code

            USAGE: [F mu s2] = npv_run(nlogpdf,theta,[nIter])

            INPUTS:
             theta - [N x D+1] initial parameter settings, where
                        N is the number of components,
                        D is the number of latent variables in the model,
                      and the last column contains the log bandwidths (variances)
              nIter (optional) - maximum number of iterations (default: 10)
              tol (optional) - change in the evidence lower bound (ELBO) for
              convergence (default: 0.0001)

            OUTPUTS:
              F - [nIter x 1] approximate ELBO value at each iteration
              mu - [N x D] component means
              s2 - [N x 1] component bandwidths
        """
        N, Dpp = theta.shape
        D      = Dpp - 1

        # set LBFGS optim arguments
        disp = 10 if verbose else None
        opts = {'disp': disp, 'maxiter': 5000,
                'gtol':1e-7, 'ftol':1e-7, 'factr':1e2}
        elbo_vals = np.zeros(niter)

        for ii in xrange(niter):
            elbo_vals[ii] = self.mc_elbo(theta)
            print "iteration %d (elbo = %2.4f)" % (ii, elbo_vals[ii])

            # first-order approximation (L1): optimize mu, one component at a time
            print " ... optimizing mus "
            for n in xrange(N):
                print " ... %d / %d " % (n, N)
                fun, gfun = self.make_elbo1_funs(theta, n)
                res = minimize(fun, x0=theta[n,:D], jac=gfun,
                                    method='L-BFGS-B', options=opts)
                theta[n,:D] = res.x

            #print theta[:,:D]
            #print " ... elbo: ", self.mc_elbo(theta)

            # second-order approximation (L2): optimize s2
            print " ... optimizing sigmas"
            mu = theta[:,:D]
            h  = np.zeros(N)
            for n in xrange(N):
                # compute Hessian trace using finite differencing or autograd
                h[n] = np.sum(np.diag(hessian(self.lnpdf)(mu[n])))

            fun, gfun = self.make_elbo2_funs(theta, h)
            res = minimize(fun, x0=theta[:,-1], jac=gfun,
                                method='L-BFGS-B', options=opts)
            theta = np.column_stack([mu, res.x])

            # calculate the approximate ELBO (L2)
            #if (ii > 1) and (np.abs(elbo_vals[ii] - elbo_vals[ii-1] < tol))
            # TODO check for convergence
            #if (ii > 1) and (np.abs(F[ii]-F[ii-1]) < tol)
            #    break # end % check for convergence

        # unpack params and return
        mu = theta[:, :D]
        s2 = np.exp(theta[:, -1]) + self.s2min
        return mu, s2, elbo_vals, theta

    def make_elbo1_funs(self, theta, n):
        """ Creates a function of mu_n that mimics equation (11) in 
        the NPVI paper

        This is a first order approximation to the ELBO that, when optimized,
        puts the means, mu_n, into areas of high probability
        """
        # model parameters
        N, Dpp = theta.shape
        D      = Dpp - 1

        # create lower bound to entropy as a function of just one n
        lbn, _ = make_lower_bound_MoGn(theta, n, s2min=1e-7)
        #lnqn = make_lnqn(theta, n, s2min=1e-7)
        #print "lnqn vs lbn: %2.4f, %2.4f"%(lnqn(theta[n,:D]), lbn(theta[n,:D]))

        def elbo1(thn):
            """ elbo with respect to mean parameter n """
            return -1.*self.lnpdf(thn)/float(N) - lbn(thn)

        return elbo1, grad(elbo1)

    def make_elbo2_funs(self, theta, h):
        """ Creates a function of lns2 that mimics equation (10) in 
        the NPVI paper

        This is a second order approximation to the ELBO that, when optimized,
        inflates the sigmas
        """
        # model parameters
        N, Dpp = theta.shape
        D      = Dpp - 1

        # create lower bound to entropy as a function of just one n
        _, lbs = make_lower_bound_MoGn(theta, n=0, s2min=1e-7)

        def elbo2(lns2):
            """ elbo with respect to mean parameter n """
            s2 = np.exp(lns2) + self.s2min
            return -.5*np.dot(s2,h) / float(N) - lbs(lns2)

        return elbo2, grad(elbo2)

    def mc_elbo(self, theta, nsamps=1000):
        """ monte carlo estimator of the ELBO """
        nsamps = 1000
        z      = mogsamples(nsamps, theta)
        lnqs   = moglogpdf(z, theta)
        llikes = self.lnpdf(z)
        return np.mean(llikes - lnqs)

    def qmean(self, theta):
        N, Dpp = theta.shape
        D      = Dpp - 1
        mu = theta[:,:D]
        return mog.mog_mean(mu, pis=np.ones(N) / float(N))

    def qcov(self, theta):
        N, Dpp = theta.shape
        D      = Dpp - 1
        mu = theta[:, :D]
        s2 = np.exp(theta[:, -1]) + self.s2min
        covs = np.array([ np.eye(D)*s for s in s2 ])
        return mog.mog_covariance(mu, covs, pis=np.ones(N)/float(N))


#####################
# utility functions #
#####################

def lower_bound_MoG(theta, s2min=1e-7,
                    return_dmu=False, n=0, return_ds2=False):
    """
    Lower bound on the entropy of a mixture of Gaussians.

    INPUT:
        theta --- all MoG parameters in the [mu; lns2] format
        s2min --- minimum variance
        return_dmu --- returns gradient with respect to mu_n, for the input n
        n --- see above
        return_ds2 ---- returns grad with respect to all s2 params
    """
    # unpack num components and dimensionality
    N, Dpp = theta.shape
    D      = Dpp-1

    # unpack mean and variance parameters
    mu   = theta[:, :D]
    s2   = np.exp(theta[:,-1]) + s2min

    # compute lower bound to entropy, Eq (7) --- we compute the 
    # Normal Probability N(mu_n | mu_j, s2_n + s2_j)
    S    = sq_dist(mu)
    s    = s2[:,None] + s2[None,:]
    lnP  = (-.5 * S / s) - .5*D*np.log(2*np.pi) - .5*D*np.log(s)
    lnqn = scpm.logsumexp(lnP, 1) - np.log(N)
    H    = np.sum(lnqn) / float(N)

    # TODO implement gradients in the same matlab style
    return -1.*H


def make_lower_bound_MoGn(theta, n, s2min=1e-7):
    """ Creates a function that computes the lower bound of a MoG as a function
    of a single mean parameter (with variances fixed)
    and as a function of all N variance parameters

    This function is autograd-able --- so grads and hessians should work
    """
    N, Dpp = theta.shape
    D      = Dpp-1

    mus = theta[:, :D]
    lns = theta[:, -1]

    def build_thmat(mu_n):
        mumat = np.row_stack([ mus[:n,:], mu_n, mus[n+1:,:] ])
        return np.column_stack([mumat, lns])

    def lbn(th_n):
        thmat = build_thmat(th_n)
        return lower_bound_MoG(thmat, s2min=s2min)

    def lbs(lns2):
        thmat = np.column_stack([mus, lns2])
        return lower_bound_MoG(thmat, s2min=s2min)

    return lbn, lbs


#def make_lnqn(theta, n, s2min=1e-7):
#    """ creates the simple function ln q_n (right below eq (7) in the paper)
#
#    Computes:
#        q_n = 1/N sum_j N( mu_n | mu_j, s_n + s_j )
#        as a function of mu_n
#
#    """
#    N, Dpp = theta.shape
#    D      = Dpp-1
#
#    # unpack
#    mus = theta[:, :D]
#    s2  = np.exp(theta[:, -1]) + s2min
#    sn  = s2[n] + s2
#
#    mus_non = np.row_stack([ mus[:n, :], mus[n+1:, :] ])
#    s_nlast = np.concatenate([ sn[:n], sn[n+1:], [sn[n]] ])
#
#    def lnqn(thn):
#
#        thmat = build_thman(thn)
#        # dist of thn to all other j = 1, ... , N
#        Sn = np.sum((thn - mus_non)**2, axis = 1)
#        S  = np.concatenate([Sn, [0.]])
#
#        # log gaussian for each component
#        lnP  = (-.5*S/s_nlast) - .5*D*np.log(2*np.pi) - .5*D*np.log(s_nlast)
#        return scpm.logsumexp(lnP) - np.log(N)
#
#    return lnqn


from autograd.util import nd
def numeric_hessian_diag(fun, th):
    D     = len(th)
    hdiag = np.zeros(D)
    gfun  = grad(fun)
    for d in xrange(D):
        de    = np.zeros(D)
        de[d] = 1e-5
        hdiag[d] = (gfun(th+de)[d] - gfun(th-de)[d]) / 2e-5
    return hdiag


def sq_dist(a, b=None):
    """ Calculate squared distance between each row to each other row
    INPUT:
        a : N x D matrix

    OUTPUT:
        N x N matrix of squared pairwise distances
    """
    b = a if b is None else b
    diff = a[:,None,:] - b[None,:,:]
    return np.sum(diff**2, axis=2)


from aip.vboost import mog
def moglogpdf(z, theta, s2min=1e-7):
    """
    Log probability according to MoG theta --- mostly for testing entropy
    bound
    """
    N, Dpp = theta.shape
    D      = Dpp - 1
    # unpack mean and variance parameters
    mu     = theta[:, :D]
    s2     = np.exp(theta[:,-1]) + s2min
    icovs  = np.array([ np.eye(D)*(1./s) for s in s2 ])
    lndets = D*np.log(s2)
    return mog.mog_logprob(z, mu, icovs, lndets, pis=np.ones(N)/float(N))


def mogsamples(nsamps, theta, s2min=1e-7):
    """
    Log probability according to MoG theta --- mostly for testing entropy
    bound
    """
    N, Dpp = theta.shape
    D      = Dpp - 1
    # unpack mean and variance parameters
    mu     = theta[:, :D]
    s2     = np.exp(theta[:,-1]) + s2min
    chols  = np.array([ np.eye(D)*np.sqrt(s) for s in s2 ])
    #lndets = D*np.log(s2)
    return mog.mog_samples(nsamps, mu, chols, pis=np.ones(N)/float(N))


if __name__=="__main__":

    # test lower bound MoG
    N, D  = 10, 2
    mu    = np.random.randn(N, D)
    lns   = np.random.randn(N)
    theta = np.column_stack([mu, lns])
    print lower_bound_MoG(theta)

    # generate samples, and compute Monte Carlo Entropy --- makes sure lower
    # bound is reasonable
    import pyprind
    Ntrials = 50
    gaps = np.zeros(Ntrials)
    for i in pyprind.prog_bar(range(Ntrials)):
        N  = np.random.randint(20) + 2
        D  = np.random.randint(20) + 2
        mu = np.random.randn(N, D)
        # test lower bound MoG
        lns   = np.random.randn(N)
        theta = np.column_stack([mu, lns])

        # compute Hmc
        nsamps = 1000
        z      = mogsamples(nsamps, theta)
        lls    = moglogpdf(z, theta)
        Hmc    = -np.mean(lls)
        Hmc_hi = Hmc + 3*np.std(lls) / np.sqrt(nsamps)

        # compute bound and store gap
        Hbound  = lower_bound_MoG(theta)
        gaps[i] = Hmc - Hbound

        # Hmc should be greater than Hbound
        assert Hmc_hi > Hbound, "bound isn't lower ya dope (%2.3f not greater than %2.3f)"%(Hmc_hi, Hbound)

    print "Gap percentiles [1, 50, 99] %s"%str(np.percentile(gaps, [1, 50, 99]))

    #########################################
    # test per mu_n function and gradient   #
    #########################################
    n   = 0
    lbn, lbs = make_lower_bound_MoGn(theta, n, s2min=1e-7)
    thn = theta[n, :D]
    assert np.isclose(lower_bound_MoG(theta), lbn(thn)), "per n is bad"
    from autograd.util import quick_grad_check, nd
    quick_grad_check(lbn, thn)

    print "Hessiandiag, numeric hessian diag"
    hlbn = hessian(lbn)
    print np.diag(hlbn(thn))

    hdiag = numeric_hessian_diag(lbn, thn)
    print hdiag

    #####################################
    # Test NVPI on a small, 2d example  #
    #####################################
    from aip.vboost import mog
    means  = np.array([ [1., 1.], [-1., -1.], [-1, 1] ])
    covs   = np.array([ 2*np.eye(2), 1*np.eye(2), 1*np.eye(2) ])
    icovs  = np.array([np.linalg.inv(c) for c in covs])
    lndets = np.array([np.linalg.slogdet(c)[1] for c in covs])
    pis    = np.ones(means.shape[0]) / float(means.shape[0])
    lnpdf  = lambda z: mog.mog_logprob(z, means, icovs, lndets, pis)
    D      = 2

    # create npvi object
    Ncomp = 5
    theta0 = np.column_stack([ 5*np.random.randn(Ncomp, D),
                               -1*np.ones(Ncomp) ])
    npvi = NPVI(lnpdf, D=2)
    mu, s2, elbo_vals, theta = npvi.run(theta0.copy(), verbose=True)
    print elbo_vals

    # plot result
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns; sns.set_style('white')
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    import autil.util.plots as pu
    pu.plot_isocontours(ax, lambda z: np.exp(lnpdf(z)), xlim=[-4, 4], ylim=[-4, 4], fill=True)
    pu.plot_isocontours(ax, lambda z: np.exp(moglogpdf(z, theta)),
                            xlim=[-4, 4], ylim=[-4, 4])

    # print lnpdf covariance
    print mog.mog_covariance(means, covs, pis)
    print npvi.qcov(theta)
