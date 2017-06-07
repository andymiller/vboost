import autograd.numpy as np
import autograd.numpy.linalg as npla
import autograd.scipy.misc as scpm

def mog_logprob(x, means, icovs, lndets, pis):
    """ compute the log likelihood according to a mixture of gaussians
        with means  = [mu0, mu1, ... muk]
             icovs  = [C0^-1, ..., CK^-1]
             lndets = ln [|C0|, ..., |CK|]
             pis    = [pi1, ..., piK] (sum to 1)
        at locations given by x = [x1, ..., xN]
    """
    xx = np.atleast_2d(x)
    D  = xx.shape[1]
    centered = xx[:,:,np.newaxis] - means.T[np.newaxis,:,:]
    solved   = np.einsum('ijk,lji->lki', icovs, centered)
    logprobs = - 0.5*np.sum(solved * centered, axis=1) - (D/2.)*np.log(2*np.pi) \
               - 0.5*lndets + np.log(pis)
    logprob  = scpm.logsumexp(logprobs, axis=1)
    if np.isscalar(x) or len(x.shape) == 1:
        return logprob[0]
    else:
        return logprob


def mog_like(x, means, icovs, lndets, pis):
    return np.exp(mog_logprob(x, means, icovs, lndets, pis))


def mog_samples(N, means, chols, pis):
    K, D = means.shape
    indices = discrete(pis, (N,))
    n_means = means[indices,:]
    n_chols = chols[indices,:,:]
    white   = npr.randn(N,D)
    color   = np.einsum('ikj,ij->ik', n_chols, white)
    return color + n_means

import autograd.numpy.random as npr
def discrete(p, shape):
    length = np.prod(shape)
    indices = p.shape[0] - \
        np.sum(npr.rand(length)[:,np.newaxis] < np.cumsum(p), axis=1)
    return indices.reshape(shape)


def mog_mean(means, pis):
    """ collapse mean --- this is a simple opp for mixtures """
    return np.dot(pis, means)


def mog_covariance(means, covs, pis):
    """ collapse out covariance --- this is slightly more complicated

    http://math.stackexchange.com/questions/195911/covariance-of-gaussian-mixtures
    """
    cov_sum   = np.sum(pis[:,None,None] * covs, axis=0)
    mu        = mog_mean(means, pis)
    centered  = means - mu[None,:]
    cov_outer = np.dot(centered.T, pis[:,None]*centered)
    return cov_sum + cov_outer


class MixtureOfGaussians():
    """Manipulate a mixture of gaussians"""

    def __init__(self, means, covs, pis):
        # dimension check
        self.K, self.D = means.shape

        # cache means, covs, pis
        self.update_params(means, covs, pis)

    def update_params(self, means, covs, pis):
        assert covs.shape[1] == covs.shape[2] == self.D
        assert self.K == covs.shape[0] == len(pis), \
                "%d != %d != %d"%(self.K, covs.shape[0], len(pis))
        assert np.isclose(np.sum(pis), 1.)
        self.means  = means
        self.covs   = covs
        self.pis    = pis
        self.lndets = np.array([npla.slogdet(c)[1] for c in self.covs])
        self.icovs  = np.array([npla.inv(c) for c in self.covs])

    def logpdf(self, x):
        return mog_logprob(x, means=self.means, icovs=self.icovs,
                           lndets=self.lndets,
                           pis  = self.pis)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self, x):
        return np.dot(self.pis, self.means)

    def var(self, x):
        return np.sum(self.covs * pis[:,None,None], axis=0)


