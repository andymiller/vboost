"""
Functions for manipulatin lists of components (collapsing into logdpf
functions, sampling, etc)
"""
from . import lowr_mvn, mog
from .misc import sigmoid, logit
import autograd.numpy as np
import autograd.scipy.misc as scpm


class LRDComponent:
    """ Functions for manipulating a low-rank plus off-diagonal component, 
    with clamped, unclamped parameters """
    def __init__(self, D, rank, lam=None,
                                clamped_m=None,
                                clamped_C=None,
                                clamped_v=None):
        self.D = D
        self.rank = rank
        self.unpack, self.setter, self.n_params = \
            make_param_unpacker(D, rank, clamped_m=clamped_m,
                                         clamped_C=clamped_C,
                                         clamped_v=clamped_v)
        self.clamped_m = clamped_m
        self.clamped_C = clamped_C
        self.clamped_v = clamped_v

        if clamped_C is not None:
            self.free_ranks = self.rank - clamped_C.shape[1]
        else:
            self.free_ranks = self.rank

        self.lam = lam

    def __str__(self):
        out = "LRDComponent<%d-dim, %d-rank"%(self.D, self.rank)
        if self.clamped_m is not None:
            out += ", clamped mean"
        if self.clamped_C is not None:
            out += ", %d clamped ranks"%self.clamped_C.shape[1]
        if self.clamped_v is not None:
            out += ", clamped spike"
        return out + ">"

    def mean(self, lam=None):
        lam = self.lam if lam is None else lam
        return self.unpack(lam)[0]

    def cov(self, lam=None):
        lam = self.lam if lam is None else lam
        _, C, v = self.unpack(lam)
        return np.dot(C, C.T) + np.diag(np.exp(v)) #np.eye(self.D) * np.exp(v)

    def icov(self, lam=None):
        lam = self.lam if lam is None else lam
        _, C, v = self.unpack(lam)
        return lowr_mvn.woodbury_invert(C, v)

    def cholesky(self, lam=None):
        lam = self.lam if lam is None else lam
        _, C, v = self.unpack(lam)
        Sig = self.cov(lam)

        # guard against numerical instability
        if not np.all(np.linalg.eigvals(Sig) > 0):
            Sig += np.eye(Sig.shape[0]) * 1e-4

        return np.linalg.cholesky(Sig)

    def lndet(self, lam=None):
        lam = self.lam if lam is None else lam
        _, C, v = self.unpack(lam)
        return lowr_mvn.woodbury_lndet(C, v)

    def sample(self, num_samps, lam=None):
        lam = self.lam if lam is None else lam
        m, C, v = self.unpack(lam)
        return lowr_mvn.mvn_lowrank_sample(num_samps, m, C, v)


def make_param_unpacker(D, rank, clamped_m = None,
                                 clamped_v = None,
                                 clamped_C = None):
    """ we parameterize (mu, C, v) as a flat vector --- this creates the
    method for unpacking the mean, and covariance parameters for this module

    Args:
      - D       : dimension of posterior
      - rank    : number of ranks to model

    Optional Args:
      - clamp_m     : D-dim array if we clamp mean (to the passed in values), 
                      otherwise None means we parameterize m
      - clamp_v     : D-dim array if we clamp log diagonal
      - clamp_ranks : number of ranks to clamp (must be less than rank)
    """
    # organize variational parameters --- mean and low rank covariance
    # be sure to account for clamped mean, covariance components, and diagonal
    start = 0
    if clamped_m is None:
        mu_slice  = slice(start, start + D)
        start    += D

    if clamped_C is None:
        C_slice   = slice(start, start + rank*D)
        start    += rank*D
    else:
        num_ranks_clamped = clamped_C.shape[1]
        num_ranks_free    = rank - num_ranks_clamped
        C_slice   = slice(start, start + num_ranks_free*D)
        start    += num_ranks_free*D

    if clamped_v is None:
        lns_slice = slice(start, start + D)
        start    += D

    n_params  = start

    # create unpack params function that is aware of the clamped values
    def unpack_params(params):
        assert len(params) == n_params, \
            "wrong shape to unpack (given %d params, needs %d params)" % \
            (len(params), n_params)
        mean = params[mu_slice] if clamped_m is None else clamped_m

        if clamped_C is None:
            C = np.reshape(params[C_slice], (D, rank))
        else:
            freeC = np.reshape(params[C_slice], (D, num_ranks_free))
            C = np.column_stack([clamped_C, freeC])

        lnsigma_sq = params[lns_slice] if clamped_v is None else clamped_v
        return mean, C, lnsigma_sq

    def set_params(params, mean=None, C=None, v=None):
        assert len(params) == n_params, "wrong shape to set"
        if mean is not None:
            assert clamped_m is None, "can't set clamped param (m)"
            assert len(mean) == (mu_slice.stop - mu_slice.start)
            params[mu_slice] = mean
        if C is not None:
            assert len(C.ravel()) == (C_slice.stop - C_slice.start)
            params[C_slice] = C.ravel()
        if v is not None:
            assert clamped_v is None, "can't set clamped param (v)"
            assert len(v) == (lns_slice.stop - lns_slice.start)
            params[lns_slice] = v
        return params

    return unpack_params, set_params, n_params


##########################################################
# functions for manipulating a list of LRD components    #
##########################################################


def comp_list_to_matrices(comp_list):
    # make sure dimensions are compatible
    dims = np.array([c.D for _, c in comp_list])
    assert np.all(dims==dims[0])

    # assemble matrices
    means  = np.array([ c.mean()     for _, c in comp_list ])
    covars = np.array([ c.cov()      for _, c in comp_list ])
    icovs  = np.array([ c.icov()     for _, c in comp_list ])
    lndets = np.array([ c.lndet()    for _, c in comp_list ])
    chols  = np.array([ c.cholesky() for _, c in comp_list ])
    pis    = np.array([ p for p, _ in comp_list ])
    return means, covars, icovs, chols, lndets, pis


def make_new_component_mixture_lnpdf(comp_list, new_rank=0):
    """ Create a mixture logpdf function that varies with new params, 
    and holds params in comp_list fixed.

    Args:
        comp_list: list of component tuples, entry i is (prob_i, params_i)

    Returns:
        lnpdf_C: log probability function for the mixture with component
                 params in comp_list (fixed).

                  lnpdf_C(x) = log( sum_i prob_i * llcomponent(x, params_i) )

        sample_C: sample from the probability distribution defined by comp_list

        lnpdf_Cplus: log probability function for the mixture with component
                     params in comp_list (fixed) AND a new component, lam_new,
                     where those params are a second argument:

                  lnpdf_Cplus(x, [lam_new; rho_new])

    """
    # assemble existing mixture in comp_list
    means, covars, icovs, chols, lndets, pis = \
        comp_list_to_matrices(comp_list)
    K, D     = means.shape
    lnpdf_C  = lambda x: mog.mog_logprob(x, means, icovs, lndets, pis)
    sample_C = lambda N: mog.mog_samples(N, means, chols, pis)

    # create new component log likelihood function
    unpacker, setter, n_params = make_param_unpacker(D, new_rank)
    def new_comp_ll(x, th):
        m, C, v = unpacker(th)
        return lowr_mvn.mvn_lowrank_logpdf(x, m, C, v)

    # create new component mixture function
    def lnpdf_Cplus(x, th, llC=None):
        """ log q(x | th_C+1), with a fixed c=1, ..., C comps
            x   : N samples (N x D matrix)
            th  : [new_component_params, logit(rho)] parameter vector
            llC : q^{(C)}(x) already computed for x
        """
        return lnpdf_Cplus_fun(x, th, lnpdf_C, new_comp_ll, llC=llC)

    return lnpdf_C, sample_C, lnpdf_Cplus, (unpacker, setter, n_params+1)


def lnpdf_Cplus_fun(x, th, lnpdf_C, new_comp_ll, llC=None):
    """ log likelihood for new component parameters
        x       : N samples (N x D matrix)
        th      : [new_component_params, logit(rho)] parameter vector
        lnpdf_C : function q^{(C)}(x) for the first C components
        new_comp_ll: function q_{C+1}(x) as a function of parameters th
        llC     : q^{(C)}(x) already computed for x
    """
    # mixing weights
    lam_new, rho_new = th[:-1], th[-1]
    prob_new = sigmoid(rho_new)
    lnpis    = np.log(np.array([1. - prob_new, prob_new]))

    # component likelihoods
    llC = lnpdf_C(x) if llC is None else llC
    llnew = new_comp_ll(x, lam_new)
    assert len(llC) == len(llnew), "passed in samples llC does not match x"

    ## sum them # (way faster than stacking and using scpm.logsumexp)
    lls = np.column_stack([ lnpis[0] + llC, lnpis[1] + llnew ])
    ll  = scpm.logsumexp(lls, axis=1)
    #ll = np.log(np.exp(llC)*(1.-prob_new) + np.exp(llnew) * prob_new)
    return ll


def matrices_to_comp_list(means, covars, pis):
    # collapse to list 
    if covars.ndim==3:
        covars = np.array([np.diag(c) for c in covars])

    # create a component list
    comp_list = []
    for k in xrange(means.shape[0]):
        lam = np.concatenate([means[k,:], .5 * np.log(covars[k,:])])
        comp_list.append( (pis[k], lam))
    return comp_list


def make_marginal(dims, comp_list):
    means, covars, _, _, _, pis = \
        comp_list_to_matrices(comp_list)
    return make_marginal_from_mats(dims, means, covars, pis)


def make_marginal_from_mats(dims, means, covars, pis):
    # select out means, covars appropriately
    means  = means[:,dims]
    covars = covars[:,dims,dims[:,None]]
    icovs  = np.array([np.linalg.inv(c) for c in covars])
    lndets = np.array([np.linalg.slogdet(c)[1] for c in covars])
    loglike = lambda x: mog.mog_logprob(x, means, icovs, lndets, pis)
    return loglike


def renormalize_comp_list(comp_list):
    pis = np.array([c[0] for c in comp_list], dtype=np.float)
    assert np.all(pis) > 0.
    pis = pis / np.sum(pis)
    return [(p, c[1]) for p, c in zip(pis, comp_list)]


def mean_comp_list(comp_list):
    means, _, _, _, _, pis = comp_list_to_matrices(comp_list)
    return mog.mog_mean(means, pis)


