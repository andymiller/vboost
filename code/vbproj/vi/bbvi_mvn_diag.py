import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian, hessian
from scipy.stats import ncx2
from vbproj.misc import mvn_diag_logpdf, mvn_diag_entropy
from .bbvi_base import BBVI


class DiagMvnBBVI(BBVI):

    def __init__(self, lnpdf, D, glnpdf=None, lnpdf_is_vectorized=False):
        """
        Implements MCVI --- exposes elbo gradient and sampling methods.
        This class breaks the gradient down into parts

        dg/dz = dlnpdf(z)/dz * dz/dlam - dlnq(z)/dz * dz/dlam - dlnq(z)/dlam

        Parameterizes with mean and log-std! (not variance!)
            lam = [mean, log-std]
        """
        # base class sets up the gradient function organization
        super(DiagMvnBBVI, self).__init__(lnpdf, D, glnpdf, lnpdf_is_vectorized)

        # we note that the second two terms, with probability one, 
        # create the vector [0, 0, 0, ..., 0, 1., 1., ..., 1.]
        self.mask = np.concatenate([np.zeros(D), np.ones(D)])
        self.num_variational_params = 2*D
        self.D = D

    #####################################################################
    # Methods for various types of gradients of the ELBO                #
    #    -- that can be plugged into FilteredOptimization routines      #
    #####################################################################

    def elbo_grad_mc(self, lam, t, n_samps=1, eps=None):
        """ monte carlo approximation of the _negative_ ELBO """
        if eps is None:
            eps = np.random.randn(n_samps, self.D)
        return -1.*np.mean(self.dlnp(lam, eps) + self.mask, axis=0)

    def nat_grad(self, lam, standard_gradient):
        finv = 1./self.fisher_info(lam)
        return finv * standard_gradient

    #############################
    # ELBO objective functions  #
    #############################

    def elbo_mc(self, lam, n_samps=100, full_monte_carlo=False):
        """ approximate the ELBO with samples """
        D = len(lam)/2
        zs  = self.sample_z(lam, n_samps=n_samps)
        if full_monte_carlo:
            elbo_vals = self.lnpdf(zs) - mvn_diag_logpdf(zs, lam[:D], lam[D:])
        else:
            elbo_vals = self.lnpdf(zs) + mvn_diag_entropy(lam[D:])
        return np.mean(elbo_vals)

    def true_elbo(self, lam, t):
        """ approximates the ELBO with 20k samples """
        return self.elbo_mc(lam, n_samps=20000)

    def sample_z(self, lam, n_samps=1, eps=None):
        """ sample from the variational distribution """
        D = self.D
        assert len(lam) == 2*D, "bad parameter length"
        if eps is None:
            eps = np.random.randn(n_samps, D)
        z = np.exp(lam[D:]) * eps + lam[None, :D]
        return z

    def callback(self, th, t, g, tskip=20, n_samps=10):
        """ custom callback --- prints statistics of all gradient comps"""
        if t % tskip == 0:
            fval = self.elbo_mc(th, n_samps=n_samps)
            gm, gv = np.abs(g[:self.D]), np.abs(g[self.D:])
            print \
"""
iter {t}; val = {val}, abs gm = {m} [{mlo}, {mhi}]
                           gv = {v} [{vlo}, {vhi}]
""".format(t=t, val="%2.4f"%fval,
                m  ="%2.4f"%np.mean(gm),
                mlo="%2.4f"%np.percentile(gm, 1.),
                mhi="%2.4f"%np.percentile(gm, 99.),
                v  ="%2.4f"%np.mean(gv),
                vlo="%2.4f"%np.percentile(gv, 1.),
                vhi="%2.4f"%np.percentile(gv, 99.))

