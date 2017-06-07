import autograd.numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats.distributions import t as tdist
import autograd.scipy.misc as scpm
from . import mog


def mixture_of_ts_samples(N, locs, scales, pis, df):
    K, D = locs.shape
    indices = mog.discrete(pis, (N,))
    n_means = locs[indices,:]
    n_chols = scales[indices,:,:]
    white   = tdist.rvs(df=df, size=(N, D))
    color   = np.einsum('ikj,ij->ik', n_chols, white)
    return color + n_means


def mixture_of_ts_logprob(x, locs, iscales, pis, df):
    xx = np.atleast_2d(x)
    D  = xx.shape[1]
    centered = xx[:,:,np.newaxis] - locs.T[np.newaxis,:,:]

    solved   = np.einsum('ijk,lji->lki', iscales, centered)
    loglikes = np.reshape(tdist.logpdf(np.reshape(solved, (-1, D)), df=df),
                          solved.shape)
    logprobs = np.sum(loglikes, axis=1) + np.log(pis)

    logprob  = scpm.logsumexp(logprobs, axis=1)
    if np.isscalar(x) or len(x.shape) == 1:
        return logprob[0]
    else:
        return logprob

