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
lnpdf, D, param_names = make_model("frisk")
output_dir = "frisk_output"

def plot_select_pairs(mcmc_df, comp_list):

    # plot a handful of bivariate marginals
    pairs = [ ['alpha_0', 'alpha_1'],
              ['alpha_0', 'beta_0'],
              ['alpha_0', 'lnsigma_a_0'],
              ['beta_0', 'beta_1'] ]

    fig, axarr = plt.subplots(1, len(pairs), figsize=(12,3))
    for ax, vpair in zip(axarr.flatten(), pairs):

        # get index for the pair
        i = mcmc_df.columns.tolist().index(vpair[0])
        j = mcmc_df.columns.tolist().index(vpair[1])
        dbidx = np.array([i, j])

        # subselect and plot MCMC samples
        sdf   = mcmc_df.values[1000:,dbidx]
        xlim = [sdf[:,0].min(), sdf[:,0].max()]
        ylim = [sdf[:,1].min(), sdf[:,1].max()]
        ax.scatter(sdf[::5,0], sdf[::5,1], c='grey', alpha=.25, s=5)

        # create marginals out of the comp_list
        mvn_marg = components.make_marginal(dbidx, comp_list)
        colors = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)
        colors = sns.color_palette("Greys", n_colors=10)
        colors = sns.light_palette("green", n_colors=8, reverse=False)
        pu.plot_isocontours(ax, lambda x: np.exp(mvn_marg(x)), xlim=xlim, ylim=ylim, colors=colors)

    return fig, axarr

if __name__=="__main__":

    #####################
    # load mcmc samples #
    #####################
    mcmc_file = os.path.join(output_dir, 'mcmc.pkl')
    with open(mcmc_file, 'rb') as f:
        nuts_dict = pickle.load(f)

    mcmc_df = pd.DataFrame(nuts_dict['chain']['th'], columns=param_names)

    ###########################
    # load in rank 3 frisk    #
    ###########################
    rank = 3
    vi_params = np.load(os.path.join(output_dir, 'initial_component-rank_%d.npy'%rank))

    # initialize LRD component
    comp     = components.LRDComponent(D, rank=rank)
    m, d, C  = vi_params[:D], vi_params[D:(2*D)], vi_params[2*D:]
    comp.lam = comp.setter(vi_params.copy(), mean=m, v=d, C=C)

    fig, axarr = plot_select_pairs(mcmc_df, [(1., comp)])
    fig.suptitle("Single rank %d component vs samples"%rank)

    ######################################
    # load in rank 3 frisk vboost comps  #
    ######################################
    with open("frisk_output/vboost_19-comp_%d-rank.pkl"%rank, "rb") as f:
        raw_comp_list = pickle.load(f)
        comp_list = [(p, components.LRDComponent(D, rank=rank, lam=c))
                     for p, c in raw_comp_list]

    fig, axarr = plot_select_pairs(mcmc_df, comp_list)
    fig.suptitle("%d rank %d components vs samples"%(len(comp_list), rank)


    ###########################################
    # first compare marginal variances/stds   #
    ###########################################
    mcmc_vars = mcmc_df.var(0)
    mcmc_mean = mcmc_df.mean(0)

    si_vars = np.diag(comp.cov())

