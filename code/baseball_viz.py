"""
Plot selected frisk marginal distributions
"""
import os
import cPickle as pickle

import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
import vbproj.misc.plots as pu
import numpy as np
import pandas as pd
from vbproj.vi.vboost import components

from mlm_main import make_model
lnpdf, D, param_names = make_model("baseball")
output_dir = "baseball_output"

#####################################
#  Plotting functions               #
#####################################

def plot_joint(ax, chain, comp_list=None, mod=None, plot_means=False):
    th_zero_kappa_marg = components.make_marginal(np.array([2, 1]), comp_list)
    import vbproj.vi.vboost.plots as pu
    pu.plot_isocontours(ax=ax, func=lambda x: np.exp(th_zero_kappa_marg(x)), xlim=[-2,.5], ylim=[1,6])
    if chain is not None:
        ax.scatter(chain['logit_theta'][1000:,0], chain['log_kappa'][1000:], alpha=.2, c='grey')
    if comp_list is not None:
        means, covars, icovs, chols, dets, pis = \
            components.comp_list_to_matrices(comp_list)
    elif mod is not None:
        means, covars, pis = mod.means_, \
                             np.array([np.diag(c) for c in mod.covars_]), \
                             mod.weights_

    if plot_means:
        ax.scatter(means[:,2], means[:,1], c='red')
        for i in xrange(means.shape[0]):
            ax.text(means[i,2], means[i,1], "comp %d, pi = %2.2f, stds = %2.2f, %2.2f" % \
                    (i, pis[i], np.sqrt(covars[i,2, 2]), np.sqrt(covars[i, 1, 1])))

    ax.set_xlabel("logit_theta_0")
    ax.set_ylabel("logit_kappa")
    return ax


def make_covariance_scatterplot(chain, qcov, ax=None, do_variance=False):
    """ Make a scatter plot comparing covariances to samples
    Args:
        chain: result from fit_baseball_mcmc (recarray w/ param names)
        qcov : covariance resulting from an approximation --- NPVI or VBoost
        ax   : figure axis

    Output:
        ax
    """
    # compute MCMC covs --- only use lower tril
    samps    = np.column_stack([chain['logit_phi'],
                                chain['log_kappa'],
                                chain['logit_theta']])
    mcmc_cov = np.tril(np.cov(samps.T), -1)
    mcmc_cov_vals = mcmc_cov[mcmc_cov != 0].flatten()
    if do_variance:
        mcmc_cov_vals = np.var(samps, axis=0)

    # compute corresponding covariance values for approx covariance
    if do_variance:
        qcov_vals = np.diag(qcov)
    else:
        qcov = np.tril(qcov, -1)
        qcov_vals = qcov[qcov != 0].flatten()

    # create axis if needed
    if ax is None:
        fig, ax = plt.figure(figsize=(6,6)), plt.gca()

    # make scatter
    ax.scatter(qcov_vals, mcmc_cov_vals)
    ax.set_ylabel("MCMC Covs", fontsize=18)
    ax.set_xlabel("q covs", fontsize=18)
    return ax


def compare_marginal(dimension, ax, chain, mfvi_comp=None, comp_list=None,
                                    means=None, covars=None, pis=None):
    if dimension == 0:
        var_name = 'logit_phi'
        s = chain[var_name][0:]
    elif dimension == 1:
        var_name = 'log_kappa'
        s = chain[var_name][0:]
    else:
        var_name = 'logit_theta'
        s = chain[var_name][0:,dimension-2]

    xgrid      = np.linspace(s.min(), s.max(), 100).reshape((-1, 1))
    ax.hist(s, bins=20, normed=True, alpha=.5)

    # MoG marginal
    if comp_list is not None:
        #ax.hist(s, 50, normed=True, alpha=.5)
        if comp_list is not None:
            marg_lnpdf = components.make_marginal(np.array([dimension]), comp_list)
        elif means is not None:
            marg_lnpdf = components.make_marginal_from_mats(np.array([dimension]), means, covars, pis)
        else:
            raise "need a marginal dude"

        ax.plot(xgrid, np.exp(marg_lnpdf(xgrid)),
                       color = sns.color_palette()[1],
                       label="VBoost (C = %d)"%len(comp_list))

    # MFVI marginal
    if mfvi_comp is not None:
        mfvi_marg_lnpdf = components.make_marginal(np.array([dimension]),
                                        comp_list = [(1., mfvi_comp)])

        ax.plot(xgrid, np.exp(mfvi_marg_lnpdf(xgrid)),
                       color = sns.color_palette()[2],
                       label="MFVI")

    ax.legend()
    ax.set_title("%s marginal"%var_name)
    return ax




if __name__=="__main__":

    ################
    # load samples #
    ################
    mcmc_file = os.path.join(output_dir, 'mcmc.pkl')
    with open(mcmc_file, 'rb') as f:
        nuts_dict = pickle.load(f)

    chain = nuts_dict['chain'][10000:]

    ####################################
    # load and plot MFVI comparison    #
    ####################################
    rank = 0
    vi_params = np.load(os.path.join(output_dir, 'initial_component-rank_%d.npy'%rank))

    # initialize LRD component
    comp     = components.LRDComponent(D, rank=rank)
    m, d, C  = vi_params[:D], vi_params[D:(2*D)], vi_params[2*D:]
    comp.lam = comp.setter(vi_params.copy(), mean=m, v=d, C=C)

    fig, ax = plt.figure(figsize=(8,6)), plt.gca()
    compare_marginal(dimension=1, ax=ax, chain=chain, mfvi_comp=comp)

    ######################################
    # load in rank 3 frisk vboost comps  #
    ######################################
    with open(os.path.join(output_dir, "vboost_19-comp_%d-rank.pkl"%rank), "rb") as f:
        raw_comp_list = pickle.load(f)
        comp_list = [(p, components.LRDComponent(D, rank=rank, lam=c))
                     for p, c in raw_comp_list]

    fig, ax = plt.figure(figsize=(8,6)), plt.gca()
    ax = plot_joint(ax, chain, comp_list)

