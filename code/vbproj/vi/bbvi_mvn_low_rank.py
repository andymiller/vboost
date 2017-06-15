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

    def __init__(self, lnpdf, D, r, **kwargs):
        """
        Normal Variational Approximation with low rank covariance structure:

            q(z; \lam) = N(m, C C^T + diag(np.exp(s)))

            lam = [m, s, C]

        note that sampling can be done ...

            z_r, z_D ~ N(0, I)
            z = m + C*z_r + exp(.5*s)*z_D

        dg/dz = dlnpdf(z)/dz * dz/dlam - dlnq(z)/dz * dz/dlam - dlnq(z)/dlam
        """
        # base class sets up the gradient function organization
        super(LowRankMvnBBVI, self).__init__(lnpdf, D, **kwargs)

        # we note that the second two terms, with probability one, 
        # create the vector [0, 0, 0, ..., 0, 1., 1., ..., 1.]
        self.mask      = np.concatenate([np.zeros(D), np.ones(D)])
        self.D, self.r = D, r

        # pack/unpack machinery
        self.construct_C, self.num_C_params, self.C_slices = \
            make_low_rank_constructor(self.D, self.r)

        #self.num_variational_params = 2*D + self.num_C_params
        self.num_variational_params = 2*D + D*r
        self.m_slice = slice(0, D)
        self.v_slice = slice(D, 2*D)
        self.C_slice = slice(2*D, self.num_variational_params)

    def unpack(self, lam):
        m, v, C = lam[self.m_slice], \
                  lam[self.v_slice], \
                  np.reshape(lam[self.C_slice], (self.D, self.r)) #self.construct_C(lam[self.C_slice])
        return m, v, C

    def pack(self, m, v, C):
        # grab tril slices
        #Clam = np.concatenate([ C[r+1:,r] for r in xrange(C.shape[1]) ])
        #return np.concatenate([m, v, Clam])
        return np.concatenate([m, v, C.ravel()])

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

    def elbo_grad_delta_approx(self, lam, t):
        """ delta method approximation of the _negative_ ELBO """
        gmu, gvar = self.delta_grad(lam)
        return -1 * gmu

    def elbo_grad_hybrid_approx(self, lam, t, rho = .5):
        """ combine a sample w/ the elbo grad mean """
        gmu, gvar = self.delta_grad(lam)
        gmu   = -1. * gmu
        gsamp = self.glnpdf_sample(lam, t)
        return (rho*gsamp + (1-rho)*gmu)

    def elbo_mc(self, lam, n_samps=100, full_monte_carlo=False):
        """ approximate the ELBO with samples """
        zs = self.sample_z(lam, n_samps=n_samps)
        m, v, C   = self.unpack(lam)
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
        z = m + np.exp(.5*v)*eps_d + np.dot(eps_r, C.T)
        return z

    #####################################################
    # internals                                         #
    #####################################################
    def dlnp(self, lam, eps):
        """ the first gradient term (data term), computes
                dlnp/dz * dz/dlambda

            If `lnpdf` is vectorized ---, eps can 

        Args:
            lam = [mean, log-std], 2xD length array
            eps = Nsamps x D matrix, for sampling randomness 
                  z_0 ~ N(0,1)
        """
        m, v, C = self.unpack(lam)
        z       = self.sample_z(lam, eps=eps)
        dlnp_dz = self.glnpdf(z)

        eps_rank, eps_diag = eps
        dm = dlnp_dz
        dv = dlnp_dz * eps_diag * np.exp(v)
        dC = dlnp_dz[:,None] * eps_rank[None,:]   # outer product
        return dm, dv, dC

    def dlnp_delta_approx(self, lam):
        """ compute the normal approximation to (dlnp_dz), 
            z    ~ N(mu, s^2) implies approximately
            g(z) ~ N(g(mu), (g'(mu) * s)^2)
        """
        D = len(lam)/2
        gmu  = self.glnpdf(lam[:D])
        gs   = self.gglnpdf(lam[:D]) * np.exp(lam[D:])
        return gmu, gs

    def delta_grad(self, lam):
        gmu, gvar = self.dlnp_dlam_approx(lam)
        return gmu + self.mask, gvar

    def dlnp_dlam_approx(self, lam):
        # compute the normal approximation to dlnp_dz)
        D = len(lam)/2
        mz, sz = self.dlnp_delta_approx(lam)

        # mean/variance for mean component (m)
        gmu_m  = mz
        gvar_m = sz**2

        # mean/variance for sigma component
        #  --- the sigma component is essentially a location scaled xi
        slam   = np.exp(lam[D:])
        gmu_s  = sz * slam            # times e^lam for the transformation
        gvar_s = (mz*mz + 3*sz*sz)*slam*slam

        # multiply this by the 
        return np.concatenate([gmu_m, gmu_s]), \
               np.concatenate([gvar_m, gvar_s])

    def nat_grad(self, lam, standard_grad):
        """ pre-multiplies the inverse fisher at lam with the passed in
        standard gradient

            F = [ Sig^{-1};     0   ;        0;
                  0       ;     F_v ;     F_vC;
                  0       ;    F_vC ;     F_C  ]


            F^{-1} = [ Sig    ;     0  ;      0;
                      0       ;     ...;    ...;
                      0       ;     ...;    ... ]

        """
        return standard_grad
        m, v, C    = self.unpack(lam)
        gm, gv, gC = self.unpack(standard_grad)

        ## the mean parameter nat grad is premultiplied by Sig
        ## this order never instantiates a DxD mat. Important! Not Sad!
        nat_gm = np.dot(C, np.dot(C.T, gm)) + np.exp(v)*gm

        ## compute approximate Inv Fish for variance parameters
        #Sig_inv_diag = lowr_mvn.woodbury_invert_diag(C, v)
        #F_v          = .5 * (Sig_inv_diag**2) * (np.exp(2*v))
        ##nat_gv = (1./F_v) * gv
        nat_gv  = .5 * gv

        ## TODO: compute approximation to F_CC, (or rather, it's inverse...)
        #F_v    = .5*(Sig_inv_diag**2)
        #nat_gC = (1./F_v[:,None]) * gC
        #nat_gC = standard_grad[self.C_slice]

        ## construct the flattened gradient and return
        return np.concatenate([nat_gm, nat_gv, gC.ravel()])


    def fisher_info(self, lam):
        """ full fisher information --- hessian of the differential
        entropy
        """
        pass


    def callback(self, th, t, g, tskip=1, n_samps=40):
        """ custom callback --- prints statistics of all gradient comps"""
        if t % tskip == 0:
            fval = self.elbo_mc(th, n_samps=n_samps)
            gm, gv, gC = self.unpack(g)
            gm, gv, gC = np.abs(gm), np.abs(gv), np.abs(gC)

            m, v, C = self.unpack(th)
            Cmags   = np.sqrt(np.sum(C**2, axis=0))

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
                C  ="%2.4f"%np.mean(gC),
                Clo="%2.4f"%np.percentile(gC, 1.),
                Chi="%2.4f"%np.percentile(gC, 99.),
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
    n_params = sl.stop

    def construct_C(lam):
        Cs = []
        for r, sl in enumerate(slices):
            Cr = np.concatenate([np.zeros(r+1), lam[sl]])
            Cs.append(Cr)
        return np.column_stack(Cs)

    return construct_C, n_params, slices

