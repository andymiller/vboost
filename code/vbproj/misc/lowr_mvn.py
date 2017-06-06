"""
Low Rank Multivariate Normal: functions to perform calculations on MVNs 
parameterized as

    x ~ N(mu, Sigma)      D-dimensional RV

    Sigma = CC^T + eye(D)*exp(v)

Functions to manipulate a 'component' --- right now specific to Gaussian
components that are either (i) diagonal or (ii) low rank + diagonal

The functions in this module typically take in the
mean (m), low rank components (C), and the log off diagonal (v)
"""
import autograd.numpy as np


def mvn_lowrank_sample(num_samps, mu, C, v, eps_lowr=None, eps_diag=None):
    """ two-step generation of a low-rank + off diagonal RV """
    # generate randomness
    D, r = np.shape(C)
    if eps_lowr is None:
        eps_lowr = np.random.randn(num_samps, r)
    if eps_diag is None:
        eps_diag = np.random.randn(num_samps, D)
    assert eps_lowr.shape[1] == r, \
        "low rank eps misaligned; C, eps_lowr shapes = %s %s"%(str(C.shape), str(eps_lowr.shape))
    assert eps_diag.shape[1] == D, \
        "diag eps misaligned; v.shape = %s, D = %d"%(str(C.shape), D)
    #print C.shape, v.shape, mu.shape, eps_diag.shape, eps_lowr.shape
    samples = np.dot(eps_lowr, C.T) + np.exp(.5*v) * eps_diag + mu #[None, :]
    return samples


ln_two_pi = 1.8378770664093453
def mvn_lowrank_logpdf(x, mean, C, s_diag):
    _, r = C.shape
    D = len(mean)
    centered = x - mean[None,:]
    # hack --- special case the rank = 0, autograd seems to have issues with
    # some dimensionless matrix manipulations
    if r != 0:
        SinvC = woodbury_solve(C, s_diag, centered.T).T
        qterm = -.5 * np.sum(centered*SinvC, axis=1)
        coef  = -.5 * (D*ln_two_pi + woodbury_lndet(C, s_diag))
        return qterm + coef
    else:
        qterm = -.5 * np.sum((x - mean)**2 / np.exp(s_diag), axis=1)
        coef  = -.5*D*np.log(2.*np.pi) - .5*np.sum(s_diag)
        return qterm + coef


ln_two_pi_e = 2.8378770664093453
def mvn_lowrank_entropy(C, lns_diag):
    """ computes the entropy with a low-rank + diagonal covariance
    directly from the low rank factors and the log-diagonal term.

    The full covariance is reconstructed

        Sigma = C C^T + np.exp(ln_diag)

    and the entropy of this normal is 

        H(Sigma) = (1/2) ln 2 pi e + (1/2) ln |det(Sigma)|

    this function uses the matrix determinant lemma to efficiently compute
    the determinant of Sigma
        ln det(Sigma) = det(s_diag + C C^T)
                      = ln det(s_diag) + ln det(I + C^T (1/s_diag) C)
    """
    D, r = np.shape(C)
    return .5 * (D*ln_two_pi_e + woodbury_lndet(C, lns_diag))


def mvn_lowrank_kl(lam_a, lam_b):
    """ https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence """
    ma, va, Ca = lam_a
    mb, vb, Cb = lam_b
    md = mb - ma
    quad_term  = np.dot(woodbury_solve(Cb, vb, md), md)
    det_term   = woodbury_lndet(Cb, vb) - woodbury_lndet(Ca, va)
    trace_term = np.sum(woodbury_invert(Cb, vb) * (np.dot(Ca, Ca.T) +
                                                   np.diag(np.exp(va))))
    return .5*(trace_term + quad_term - len(ma) + det_term)


def mvn_kl_div(mu_a, Sig_a, mu_b, Sig_b):
    """ computes kl dv KL(a || b) 
    using answer from:
    http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    TODO: exploit th elow rank structure here --- this is general
    """
    _, lndet_a = np.linalg.slogdet(Sig_a)
    _, lndet_b = np.linalg.slogdet(Sig_b)
    lndet_rat = lndet_b - lndet_a
    trace_term = np.trace(np.linalg.solve(Sig_b, Sig_a))
    quad_term  = np.dot(mu_b - mu_a, np.linalg.solve(Sig_b, mu_b - mu_a))
    return .5 * (lndet_rat - len(mu_a) + trace_term + quad_term)


def mvn_sym_kl_div(mu_a, Sig_a, mu_b, Sig_b):
    return mvn_kl_div(mu_a, Sig_a, mu_b, Sig_b) + \
           mvn_kl_div(mu_b, Sig_b, mu_a, Sig_a)


def mvn_lowrank_params(mean, C, s_diag, marg_idx=None):
    """ constructs mean and sigma (for all or some subset of marginals) """
    marg_idx = np.arange(len(mean)) if marg_idx is None else marg_idx
    Sigma = np.dot(C[marg_idx,:], C[marg_idx,:].T) + \
            np.eye(len(marg_idx)) * np.exp(s_diag[marg_idx])
    return mean[marg_idx], Sigma


from autograd.scipy.stats import multivariate_normal as mvn
def make_mvn_lowrank_marginal(mean, C, s_diag, marg_idx):
    mu_marg, Sigma_marg = mvn_lowrank_params(mean, C, s_diag, marg_idx)
    return lambda x: mvn.logpdf(x, mean=mu_marg, cov=Sigma_marg,
                                   allow_singular=True)

def fisher_info(params):
    """ returns the fisher information matrix (diagonal) for a multivariate
    normal distribution with params = [mu, ln sigma] """
    raise NotImplementedError()
    from autograd import hessian
    #D = len(params) / 2
    #mean, log_std = params[:D], params[D:]
    #return np.concatenate([np.exp(-2.*log_std),
    #                       2*np.ones(D)])


def standard_to_natural(mu, C, v):
    """ efficiently converts standard (low rank) parameters to natural 
    Based on exponential family wikipedia parametrization:
        https://en.wikipedia.org/wiki/Exponential_family
    """
    Sinv = woodbury_invert(C, v)
    return np.dot(Sinv, mu), -.5*Sinv


def log_partition(mu, C, v):
    """ compute log partition as a function of natural parameters"""
    Sinv = woodbury_invert(C, v)
    return .5 * (np.dot(np.dot(mu, Sinv), mu) + woodbury_lndet(C, v))


def woodbury_invert(C, v):
    """ returns (diag(exp(a)) + UV)^{-1} """
    D, r   = np.shape(C)
    inv_v  = np.exp(-v)
    aC     = C*inv_v[:,None]
    r_term = np.eye(r) + np.dot(C.T, aC)
    Sinv   = np.diag(inv_v) - np.dot(aC, np.linalg.solve(r_term, aC.T))
    return Sinv


def woodbury_invert_diag(C, v):
    """ returns the diagonal of (CC^t + diag(exp(v)))^{-1} 
        without creating a big DxD matrix (only instantiates a Dxr)
    """
    D, r   = np.shape(C)
    inv_v  = np.exp(-v)
    aC     = C*inv_v[:,None]
    r_term = np.eye(r) + np.dot(C.T, aC)
    Cr_inv = np.linalg.solve(r_term, aC.T)
    Sinv_diag = inv_v - np.sum(aC*Cr_inv.T, axis=1)
    return Sinv_diag


def woodbury_solve(C, v, p):
    """ Computes the matrix vector product (Sigma)^{-1} p
    where

        Sigma = CCt + diag(exp(a))
        C     = D x r real valued matrix
        v     = D dimensional real valued vector

    The point of this function is that you never have to explicitly
    represent the full DxD matrix to do this multiplication --- hopefully
    that will cut down on memory allocations, allow for better scaling

    in comments below, we write Sigma = CCt + A, where A = diag(exp(v))
    """
    if p.ndim == 1:
        p = p[:,None]

    assert C.ndim == 2
    D, r      = np.shape(C)
    inv_v     = np.exp(-v)                         # A^{-1}
    aC        = C*inv_v[:, None]                   # A^{-1} C
    r_term    = np.eye(r) + np.dot(C.T, aC)
    inv_term  = np.linalg.solve(r_term, aC.T)

    # multiply the inverse against the input vectors p = (N vectors of dim D)
    back_term = np.dot(aC.T, p)
    ainvp     = inv_v[:,None] * p
    bterm     = np.dot(inv_term.T, back_term)
    solved = np.squeeze(ainvp - bterm)
    return solved


def woodbury_lndet(C, v):
    """ returns |det(Sigma)| = |det(CC^T + exp(v))| """
    D, r = np.shape(C)
    diag_lndet = np.sum(v)
    if r == 0:
        lowr_lndet = 0
    else:
        sgn, lowr_lndet = \
            np.linalg.slogdet(np.eye(r) +
                              np.dot(C.T, C*np.exp(-v)[:,None]))
        assert sgn > 0., "bad C, v"
    return diag_lndet + lowr_lndet


def woodbury_solve_vec(C, v, p):
    """ Vectorzed woodbury solve --- overkill
    Computes the matrix vector product (Sigma)^{-1} p
    where

        Sigma = CCt + diag(exp(a))
        C     = D x r real valued matrix
        v     = D dimensional real valued vector

    The point of this function is that you never have to explicitly
    represent the full DxD matrix to do this multiplication --- hopefully
    that will cut down on memory allocations, allow for better scaling

    in comments below, we write Sigma = CCt + A, where A = diag(exp(v))
    """
    # set up vectorization
    if C.ndim == 2:
        C = np.expand_dims(C, 0)
        assert v.ndim == 1, "v shape mismatched"
        assert p.ndim == 1, "p shape mismatched"
        v = np.expand_dims(v, 0)
        p = np.expand_dims(p, 0)

    bsize, D, r = np.shape(C)

    # compute the inverse of the digaonal copmonent
    inv_v  = np.exp(-v)                         # A^{-1}
    aC     = C*inv_v[:, :, None]                # A^{-1} C

    # low rank, r x r term: (Ir + Ct A^{-1} C)
    r_term = np.einsum('ijk,ijh->ikh', C, aC) + \
             np.eye(r)

    # compute inverse term (broadcasts over first axis) 
    #  (Ir + Ct A^{-1} C)^{-1} (Ct A^{-1})
    # in einsum notation:
    #   - i indexes minibatch (vectorization)
    #   - r indexes rank dimension
    #   - d indexes D dimension (obs dimension)
    inv_term  = np.linalg.solve(r_term, np.swapaxes(aC, 1, 2))
    back_term = np.einsum('idr,id->ir', aC, p) # (Ct A^{-1} p)
    Sigvs = inv_v*p - np.einsum('ird,ir->id', inv_term, back_term)
    return Sigvs


if __name__=="__main__":

    # Test woodbury
    C = np.random.randn(10, 3)
    v = np.random.randn(10)*2
    Sigma = np.dot(C, C.T) + np.diag(np.exp(v))
    Sinv = np.linalg.inv(Sigma) 
    Sinv_wood = woodbury_invert(C, v)
    assert np.allclose(Sinv, Sinv_wood, atol=1e-6), "woodbury!"

    _, lndet = np.linalg.slogdet(Sigma)
    lndet_wood = woodbury_lndet(C, v)
    assert np.allclose(lndet, lndet_wood), "woodbury det!"

    # test woodbury solve
    p = np.random.randn(10)
    a_wood = woodbury_solve(C, v, p)
    a = np.dot(Sinv, p)
    assert np.allclose(a, a_wood), "woodbury solve!"

    p = np.random.randn(10, 23)
    aw = woodbury_solve(C, v, p)
    aa = np.dot(Sinv, p)
    assert np.allclose(aw, aa), "woodbury solve vectorized!"

    # test vectorized version of woodbury solve --- stack of C's, vs and ps
    bsize, D, r = 11, 10, 2
    C = np.random.randn(bsize, D, r)
    v = np.random.randn(bsize, D)
    p = np.random.randn(bsize, D)
    res = woodbury_solve_vec(C, v, p)

    sigs = np.array([ np.dot(CC, CC.T) + np.diag(np.exp(vv))
                      for CC, vv in zip(C, v) ])
    res0 = np.linalg.solve(sigs, p)
    assert np.allclose(res, res0), "woodubry vectorized solve!"

    # test log pdf
    D, rank = 10, 0
    m = np.random.randn(D)
    C = np.random.randn(D, rank)
    v = np.random.randn(D)
    x = mvn_lowrank_sample(100, m, C, v)
    llwood = mvn_lowrank_logpdf(x, m, C, v)
    from scipy.stats import multivariate_normal as mvn
    ll = mvn.logpdf(x, mean=m, cov= np.dot(C, C.T) + np.eye(D)*np.exp(v))
    assert np.allclose(llwood, ll), "woodbury mvn loglike"

    # test covariance
    D, rank = 10, 2
    m = np.random.randn(D)
    C = np.random.randn(D, rank)
    v = np.random.randn(D)
    x = mvn_lowrank_sample(1e6, m, C, v)

    S = np.dot(C, C.T) + np.diag(np.exp(v))
    Shat = np.cov(x.T)
    np.mean(np.abs(S-Shat))
    np.allclose(S, Shat, rtol=.05)

    # test inversion diag
    #from aip.misc.lowr_mvn import woodbury_invert, woodbury_invert_diag
    Sinv = woodbury_invert(C, v)
    Sinv_diag = woodbury_invert_diag(C, v)
    assert np.allclose(np.diag(Sinv), Sinv_diag)

