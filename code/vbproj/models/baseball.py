import autograd.numpy as np
import autograd.numpy.random as npr
import pandas as pd
import os


#################################################
# Multi-level model for binary outcomes         #
#################################################
data_dir = os.path.dirname(__file__)
df = pd.read_table(os.path.join(data_dir, 'data/baseball/efron-morris-75-data.tsv'))
df['RemainingHits'] = df['SeasonHits'] - df['Hits']
df.head()

data_dict = {'N': df.shape[0],
             'K': df['At-Bats'].values,
             'y': df['Hits'].values,
             'K_new': df['RemainingAt-Bats'].values,
             'y_new': df['RemainingHits'].values}

# STAN Model (for sanity checking!)
stan_data = \
"""
data {
  int<lower=0> N;           // items
  int<lower=0> K[N];        // initial trials
  int<lower=0> y[N];        // initial successes

  int<lower=0> K_new[N];    // new trials
  int<lower=0> y_new[N];    // new successes
}
"""

stan_params = \
"""
parameters {
  real<lower=0, upper=1> phi;         // population chance of success
  real<lower=1> kappa;                // population concentration
  vector<lower=0, upper=1>[N] theta;  // chance of success 
}
"""

stan_model = \
"""
model {
  kappa ~ pareto(1, 1.5);                        // hyperprior
  theta ~ beta(phi * kappa, (1 - phi) * kappa);  // prior
  y ~ binomial(K, theta);                        // likelihood
}
"""

# equivalent numpy model
from autograd.scipy.special import gammaln
def pareto_lnpdf(y, ymin, alpha):
    num   = np.log(alpha) + alpha * np.log(ymin)
    denom = (alpha+1) * np.log(y)
    return num - denom

def beta_lnpdf(y, alpha, beta):
    """ y is either N x 1 or N x D,
        alpha is len(N) and beta is len(D)
    """
    num   = np.log(y)*(alpha-1)[:,None] + np.log(1-y)*(beta-1)[:,None]
    denom = gammaln(alpha) + gammaln(beta) - gammaln(alpha+beta)
    return num - denom[:,None]

def binom_lnpmf(n, N, theta):
    return gammaln(N+1) - gammaln(n+1) - gammaln(N-n+1) + \
           n*np.log(theta) + (N-n)*np.log(1.-theta)

def logit(a):
    return np.log(a) - np.log(1.-a)

def sigmoid(a):
    return 1./(1. + np.exp(-a))

def counted(fn):
    def wrapper(*args, **kwargs):
        if np.isscalar(args[0]):
            wrapper.called += 1
        else:
            wrapper.called += len(args[0])
        return fn(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__= fn.__name__
    return wrapper


@counted
def lnpdf(logit_phi, log_kappa, logit_theta, y, K):
    # transform params
    phi   = sigmoid(logit_phi)
    kappa = np.exp(log_kappa)
    theta = sigmoid(logit_theta)

    # priors
    prior_kappa = pareto_lnpdf(kappa, 1, 1.5)
    prior_theta = beta_lnpdf(theta, phi*kappa, (1-phi)*kappa)
    ll_y = binom_lnpmf(y, K, theta)

    # return the correct dimensions
    if ll_y.ndim==1:
        return np.sum(ll_y) + prior_kappa + np.sum(prior_theta)
    return np.sum(ll_y, axis=1) + prior_kappa + np.sum(prior_theta, axis=1)


# data implicit
def lnpdf_flat(th, i):
    th = np.atleast_2d(th)
    logit_phi, log_kappa, logit_theta = th[:,0], th[:,1], th[:,2:]
    return lnpdf(logit_phi, log_kappa, logit_theta,
                 y=data_dict['y'], K=data_dict['K'])


def lnp(logit_phi, log_kappa, logit_theta):
    return lnpdf(logit_phi, log_kappa, logit_theta,
                 data_dict['y'], data_dict['K'])

# posterior dimension
D = df.shape[0] + 2


if __name__=="__main__":

    import pystan
    mod = pystan.stan(model_code=stan_data + stan_params + stan_model,
                      data = data_dict,
                      iter  =5000,
                      chains=4)
    samps = mod.extract()
    sm    = pystan.StanModel(model_code=stan_data + stan_params + stan_model)
    sm.optimizing(data=data_dict)

