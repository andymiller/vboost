"""
Class for experimenting with gradient models for low rank + diag
variational approximations
"""
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian
from scipy.stats import ncx2
from vbproj.misc import mvn_diag_logpdf
from vbproj.misc import lowr_mvn
from .bbvi_base import BBVI


class LowRankMvnBBVI(BBVI):

    def __init__(self, lnpdf, D, r, glnpdf=None, lnpdf_is_vectorized=False):
        """
        Normal Variational Approximation with low rank covariance structure:

            q(z; \lam) = N(m, C C^T + diag(np.exp(s)))

            lam = [m, s, C]

        note that sampling can be done ...

            z_r, z_D ~ N(0, I)
            z = m + C*z_r + exp(.5*s)*z_D

        dg/dz = dlnpdf(z)/dz * dz/dlam - dlnq(z)/dz * dz/dlam - dlnq(z)/dlam
        
        Args:
            - lnpdf: autograd-able function handle (unnormalized log prob over latent var z)
            - D    : dimensionality of latent var z
            - r    : rank of this approximation (between 0 and D)
            - glnpdf : grad of lnpdf (not implemented)
            - lnpdf_is_vectorized : True if 
        """
        super(LowRankMvnBBVI, self).__init__(lnpdf, D, glnpdf, lnpdf_is_vectorized)
        self.D, self.r = D, r

        # flat vector unpacking
        self.num_variational_params = 2*D + D*r
        self.m_slice = slice(0, D)
        self.v_slice = slice(D, 2*D)
        self.C_slice = slice(2*D, self.num_variational_params)

        # TODO --- compare to identified low rank cholesky parameterization
        # create the vector [0, 0, 0, ..., 0, 1., 1., ..., 1.]
        #self.mask      = np.concatenate([np.zeros(D), np.ones(D)])
        # pack/unpack machinery
        #self.construct_C, self.num_C_params, self.C_slices = \
        #    make_low_rank_constructor(self.D, self.r)
        #self.num_variational_params = 2*D + self.num_C_params

    def unpack(self, lam):
        m, v, C = lam[self.m_slice], \
                  lam[self.v_slice], \
                  np.reshape(lam[self.C_slice], (self.D, self.r)) #self.construct_C(lam[self.C_slice])
        return m, v, C

    def pack(self, m, v, C):
        return np.concatenate([m, v, np.ravel(C)])

    #############################################################
    # Elbo and elbo gradient methods                            #
    #############################################################

    def elbo_grad_mc(self, lam, t, n_samps=1, eps=None):
        """ monte carlo approximation of the _negative_ ELBO """
        if eps is None:
            eps = (np.random.randn(n_samps, self.r), \
                   np.random.randn(n_samps, self.D))
        # TODO update this to directly use grad
        return -1.*grad(self.elbo_mc)(lam, n_samps=n_samps)
        #return -1.*np.mean(self.dlnp(lam, eps) + self.mask, axis=0)

    def elbo_mc(self, lam, n_samps=100, full_monte_carlo=False):
        """ approximate the ELBO with samples """
        zs = self.sample_z(lam, n_samps=n_samps)
        m, v, C = self.unpack(lam)
        if full_monte_carlo:
            elbo_vals = self.lnpdf(zs) - \
                lowr_mvn.mvn_lowrank_logpdf(zs, m, C, v)
        else:
            elbo_vals = self.lnpdf(zs) + lowr_mvn.mvn_lowrank_entropy(C, v)
        return np.mean(elbo_vals)

    def true_elbo(self, lam, t):
        """ approximates the ELBO with 20k samples """
        return self.elbo_mc(lam, n_samps=20000)

    def sample_z(self, lam, n_samps=1, eps=None):
        """
        eps is a tuple, (eps_rank, eps_diag)
        """
        if eps is None:
            eps_r, eps_d = np.random.randn(n_samps, self.r), \
                           np.random.randn(n_samps, self.D)
        else:
            eps_r, eps_d = eps  # eps is a tuple here ---
        m, v, C = self.unpack(lam)
        if self.r > 0:
            z = m + np.exp(.5*v)*eps_d + np.dot(eps_r, C.T)
        else:
            z = m + np.exp(.5*v)*eps_d
        return z

    def callback(self, th, t, g, tskip=20, n_samps=100):
        """ custom callback --- prints statistics of all gradient comps"""
        if t % tskip == 0:
            fval = self.elbo_mc(th, n_samps=n_samps)
            gm, gv, gC = self.unpack(g)
            gm, gv, gC = np.abs(gm), np.abs(gv), np.abs(gC)

            m, v, C = self.unpack(th)
            Cmags   = np.sqrt(np.sum(C**2, axis=0))

            if self.r > 0:
                Cm ="%2.4f"%np.mean(gC),
                Clo="%2.4f"%np.percentile(gC, 1.),
                Chi="%2.4f"%np.percentile(gC, 99.),
            else:
                 Cm, Clo, Chi = "na", "na", "na"

            print \
"""
iter {t}; val = {val},
          abs gm         = {m} [{mlo}, {mhi}]
          gv             = {v} [{vlo}, {vhi}]
          gC ({D} x {r}) = {C} [{Clo}, {Chi}]
          Comp mags      = {Cmags} 

""".format(t=t, val="%2.4f"%fval,
                D  = "%d"%self.D, r="%d"%self.r,
                m  ="%2.4f"%np.mean(gm),
                mlo="%2.4f"%np.percentile(gm, 1.),
                mhi="%2.4f"%np.percentile(gm, 99.),
                v  ="%2.4f"%np.mean(gv),
                vlo="%2.4f"%np.percentile(gv, 1.),
                vhi="%2.4f"%np.percentile(gv, 99.),
                C  =Cm, Clo=Clo, Chi=Chi,
                Cmags=np.str(Cmags))


###################################################
# helper function to manipulate parameterizations #
###################################################

def make_low_rank_constructor(D, rank):
    """ creates a function that takes in a flat parameter vector (lam), and
    outputs a D x rank matrix such that (for D, and rank=3)

        C = [ 0        0         0
              lam_1    0         0
              lam_2    lam_D     0
              ...      ...       lam_2D-1
              ...      ...       ...
              lam_D-1  lam_2D-2  lam_3D-3 ]

    """
    start  = 0
    slices = []
    for r in xrange(rank):
        sl    = slice(start, start+D-1-r)
        start = sl.stop
        slices.append(sl)

    # num params that go into this C matrix
    n_params = slices[-1].stop

    def construct_C(lam):
        Cs = []
        for r, sl in enumerate(slices):
            Cr = np.concatenate([np.zeros(r+1), lam[sl]])
            Cs.append(Cr)
        return np.column_stack(Cs)

    return construct_C, n_params, slices

