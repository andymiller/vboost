import autograd.numpy as np
from autograd.scipy.special import gammaln

def sigmoid(a):
    return 1. / (1. + np.exp(-a))


def logit(a):
    return np.log(a) - np.log(1-a)


def mvn_diag_logpdf(x, mean, log_std):
    D = len(mean)
    qterm = -.5 * np.sum((x - mean)**2 / np.exp(2.*log_std), axis=1)
    coef  = -.5*D * np.log(2.*np.pi) - np.sum(log_std)
    return qterm + coef

def mvn_diag_entropy(log_std):
    D = len(log_std)
    return .5 * (D*np.log(2*np.pi) + np.sum(log_std))

def mvn_logpdf(x, mean, icholSigma):
    D     = len(mean)
    coef  = -.5*D*np.log(2.*np.pi)
    dterm = np.sum(np.log(np.diag(icholSigma)))
    white = np.dot(np.atleast_2d(x) - mean, icholSigma.T)
    qterm = -.5*np.sum(white**2, axis=1)
    ll = coef + dterm + qterm
    if len(ll) == 1:
        return ll[0]
    return ll


def mvn_fisher_info(params):
    """ returns the fisher information matrix (diagonal) for a multivariate
    normal distribution with params = [mu, ln sigma] """
    D = len(params) / 2
    mean, log_std = params[:D], params[D:]
    return np.concatenate([np.exp(-2.*log_std),
                           2*np.ones(D)])


def gamma_lnpdf(x, shape, rate):
    """ shape/rate formulation on wikipedia """
    coef  = shape * np.log(rate) - gammaln(shape)
    dterm = (shape-1.) * np.log(x) - rate*x
    return coef + dterm


def make_fixed_cov_mvn_logpdf(Sigma):
    icholSigma = np.linalg.inv(np.linalg.cholesky(Sigma))
    return lambda x, mean: mvn_logpdf(x, mean, icholSigma)


def unpack_params(params):
    mean, log_std = np.split(params, 2)
    return mean, log_std


def unconstrained_to_simplex(rhos):
    rhosf = np.concatenate([rhos, [0.]])
    pis   = np.exp(rhosf) / np.sum(np.exp(rhosf))
    return pis


def simplex_to_unconstrained(pis):
    lnpis = np.log(pis)
    return (lnpis - lnpis[-1])[:-1]

