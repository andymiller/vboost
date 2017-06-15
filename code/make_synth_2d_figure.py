"""
Make panels for the Toy 2-d Example Figure
"""
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from vbproj.vi.vboost import mog_bbvi, mog, plots, components
from vbproj import vi
from vbproj.gsmooth.opt import FilteredOptimization
from vbproj.gsmooth.smoothers import AdamFilter
from vbproj.vi.vboost.components import LRDComponent

##################################################
# import matplotlib with TrueType embedded fonts #
##################################################
import matplotlib
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
import os

# for figure output
OUTPUT_DIR = 'figures/synth_2d_fig/'
if not os.path.exists(OUTPUT_DIR):
    print "  saving plots to ", OUTPUT_DIR
    os.makedirs(OUTPUT_DIR)


def gen_target_dist(D = 2, means=None, covars=None, pis=None):
    """ mixture of gaussians """
    K      = 2
    if means is None:
        means  = np.random.randn(K, D) * 2
    if covars is None:
        As     = np.random.randn(K, D, D)
        covars = np.array([ np.eye(D)*.1 + np.dot(A, A.T) for A in As ])
    icovs  = np.array([np.linalg.inv(c) for c in covars])
    chols  = np.array([np.linalg.cholesky(c) for c in covars])
    dets   = np.array([np.linalg.det(c) for c in covars])
    if pis is None:
        pis    = np.ones(K) / float(K)
    lnpdf  = lambda x: mog.mog_logprob(x, means, icovs, dets, pis)
    lnpdft = lambda x, t: mog.mog_logprob(x, means, icovs, dets, pis)
    sample = lambda N: mog.mog_samples(N, means, chols, pis)
    return lnpdf, lnpdft, sample


# single component initialization
def mfvi_init(lnpdf, D, num_iters):
    mfvi = vi.DiagMvnBBVI(lnpdf, D, lnpdf_is_vectorized=True)
    elbo_grad = grad(mfvi.elbo_mc)
    def mc_grad_fun(lam, t):
        return -1.*elbo_grad(lam, n_samps=1024) #args.vboost_nsamps)

    lam = np.random.randn(mfvi.num_variational_params) * .01
    lam[D:] = -1.
    mc_opt = FilteredOptimization(
                  mc_grad_fun,
                  lam.copy(),
                  save_grads = False,
                  grad_filter = AdamFilter(),
                  fun = lambda lam, t: mfvi.elbo_mc(lam, n_samps=1000),
                  callback=mfvi.callback)
    mc_opt.run(num_iters=num_iters, step_size=.05)
    return mc_opt.params.copy()


if __name__=="__main__":

    ########################
    # gen toy distribution #
    ########################
    np.random.seed(42)
    D      = 2
    K      = 2
    means  = np.array([[-1, 1], [1, -1]])
    covars = .5*np.array([np.eye(D) for _ in xrange(2)])
    lnpdf, lnpdft, sample = gen_target_dist(D, means=means, covars=covars)
    xlim, ylim = [-3, 3], [-3, 3]

    ###############################################
    # BASE CASE: fit a single gaussian using BBVI #
    ###############################################
    mfvi_lam  = mfvi_init(lnpdf, D, num_iters=400)
    mfvi_comp = LRDComponent(D, rank=0, lam=mfvi_lam.copy())

    #####################
    # VBOOST Iterations #
    #####################
    # Variational Boosting Object (with initial component MFVI component)
    vbobj = vi.MixtureVI(lambda z, t: lnpdf(z),
                         D                     = D,
                         comp_list             = [(1., mfvi_comp)],
                         fix_component_samples = True,
                         break_condition='percent')

    for k in xrange(20):

        # plot with components
        fname   = os.path.join(OUTPUT_DIR, "n_comp_%d.pdf"%k)
        fig, ax = plt.figure(figsize=(8,6)), plt.gca()
        plots.plot_isocontours(ax, lambda x: np.exp(lnpdf(x)),
                               xlim=xlim, ylim=ylim, fill=True)
        qprob_C, _, _, _ = \
            components.make_new_component_mixture_lnpdf(vbobj.comp_list)
        plots.plot_isocontours(ax, lambda x: np.exp(qprob_C(x)),
                               xlim=xlim, ylim=ylim, colors='darkred')
        plt.savefig(fname, bbox_inches='tight')
        plt.close("all")

        print "\n\n=========== iter %d ============="%k
        # initialize new comp w/ weighted EM scheme
        (init_prob, init_comp) = \
            vbobj.fit_mvn_comp_iw_em(new_rank = mfvi_comp.rank,
                                     num_samples=200,
                                     importance_dist = 'gauss-mixture',
                                     use_max_sample=False)
        init_prob = np.max([init_prob, .5])

        # fit new component
        vbobj.fit_new_comp(init_comp = init_comp,
                           init_prob = init_prob,
                           max_iter  = 1000,
                           step_size = .1,
                           num_new_component_samples    = 200, #10*D,
                           num_previous_mixture_samples = 200, #*D,
                           fix_component_samples=True,
                           gradient_type="standard", #component_approx_static_rho",
                           break_condition='percent')

        comp_list = mog_bbvi.fit_mixture_weights(
			vbobj.comp_list, vbobj.lnpdf,
                        num_iters=1000, step_size=.1,
                        num_samps_per_component=10*D,
                        ax=None)
        vbobj.comp_list = comp_list

