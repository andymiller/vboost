"""
Make 1-d toy figure
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

OUTPUT_DIR = 'figures/vb_cartoon'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def mfvi_init(lnpdf, D, num_iters):
    mfvi = vi.DiagMvnBBVI(lnpdf, D, lnpdf_is_vectorized=True)
    elbo_grad = grad(mfvi.elbo_mc)
    def mc_grad_fun(lam, t):
        return -1.*elbo_grad(lam, n_samps=1024) #args.vboost_nsamps)

    lam = np.random.randn(mfvi.num_variational_params) * .001
    lam[D:] = -3.
    mc_opt = FilteredOptimization(
                  mc_grad_fun,
                  lam.copy(),
                  save_grads = False,
                  grad_filter = AdamFilter(),
                  fun = lambda lam, t: mfvi.elbo_mc(lam, n_samps=1000),
                  callback=mfvi.callback)
    mc_opt.run(num_iters=num_iters, step_size=.1)
    return mc_opt.params.copy()


def plot_two_comp(ax, comp_list, comp_two_text="initial new comp", scale=.5):
    pis = np.array([c[0] for c in comp_list])
    qlnpdf1, _, _, _ = \
        components.make_new_component_mixture_lnpdf([(1., comp_list[0][1])])

    qlnpdf2, _, _, _ = \
        components.make_new_component_mixture_lnpdf([(1., comp_list[1][1])])
    ax.fill_between(xgrid.flatten(), 0., np.exp(lnpdf(xgrid)).flatten(),
                           color='grey', alpha=.25, label="target")
    ax.plot(xgrid, scale*pis[0]*np.exp(qlnpdf1(xgrid)),
                   c=sns.color_palette()[0], label="existing approx",
                   linewidth=4)
    ax.plot(xgrid, scale*pis[1]*np.exp(qlnpdf2(xgrid)), "--",
                   c=sns.color_palette()[2], label=comp_two_text,
                   linewidth=4)
    plt.axis('off')
    ax.legend(loc='best', fontsize=20)


if __name__=="__main__":
    np.random.seed(42)

    #########################
    # create 1-d toy target #
    #########################
    means  = np.array([[-3.], [2.5]])
    covars = .5*np.array([[[2]], [[6]]])
    icovs  = 1./covars
    dets   = np.array([np.linalg.det(c) for c in covars])
    pis    = np.array([.65, .35])
    lnpdf  = lambda x: mog.mog_logprob(x, means, icovs, dets, pis) #+ np.log(1/.75)
    D = 1

    #######################
    # mean field approx   #
    #######################
    mfvi_lam  = mfvi_init(lnpdf, D, num_iters=1000)
    mfvi_comp = LRDComponent(D, rank=0, lam=mfvi_lam.copy())

    #############
    # plot MFVI #
    #############
    qmeans, qcovars, qicovs, qchols, qlndets, qpis = \
        components.comp_list_to_matrices([(1., mfvi_comp)])

    #qmu, qlnstd = mfvi_lam
    #qmeans = np.array([[qmu]])
    #qcovars = np.array([[[np.exp(2.*qlnstd)]]])
    #qicovs  = 1./qcovars
    #qdets    = np.array([np.linalg.det(c) for c in qcovars])
    #qpis     = np.array([1.])
    qlnpdf = lambda x, i: mog.mog_logprob(x, qmeans, qicovs, np.exp(qlndets), qpis)

    fig, ax = plt.figure(figsize=(15,4)), plt.gca()
    xgrid = np.linspace(-6.5, 6.5, 100).reshape((-1, 1))
    ax.fill_between(xgrid.flatten(), 0., np.exp(lnpdf(xgrid)).flatten(),
                    color='grey', alpha=.25, label="target")
    ax.plot(xgrid, .75*np.exp(qlnpdf(xgrid, 0)),  # hack to make the heights line up
                   c=sns.color_palette()[0], label="existing approx", linewidth=4)
    ax.legend(loc='best', fontsize=20)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+.01)
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "single_mode.pdf"), bbox_inches='tight')

    ##################################
    # create vboost object           #
    ##################################
    vbobj     = vi.MixtureVI(lambda z, t: lnpdf(z),
                             D                     = D,
                             comp_list             = [(1., mfvi_comp)],
                             fix_component_samples = True,
                             break_condition='percent')

    ###############################################
    # use max sample to initialize new component  #
    ###############################################
    from scipy.stats.distributions import t as tdist
    qm, qs = qmeans[0,0], np.sqrt(qcovars[0,0])
    tsamps = tdist.rvs(df=10, loc=qm, scale=qs, size=1000)
    tlls   = tdist.logpdf(tsamps, df=10, loc=qm, scale=qs)
    plls   = lnpdf(tsamps[:,None])
    lnws   = plls - tlls
    init_lam  = np.array([tsamps[lnws.argmax()], mfvi_lam[1]])
    init_comp = LRDComponent(D, rank=0, lam=init_lam)
    print "init component mean: ", init_comp.mean()

    # plot initial component
    fig, ax = plt.figure(figsize=(15,4)), plt.gca()
    plot_two_comp(ax, [(.7, mfvi_comp), (.3, init_comp)], scale=.6)
    plt.savefig(os.path.join(OUTPUT_DIR, "mode_two_initial.pdf"), bbox_inches='tight')

    ################################################
    # Optimize new objective to tighten component  #
    ################################################
    vbobj.fit_new_comp(
            init_comp                    = init_comp,
            init_prob                    = .1, #None, #init_prob,
            max_iter                     = 500,
            step_size                    = .01,
            num_new_component_samples    = 100,
            num_previous_mixture_samples = 100,
            fix_component_samples        = True, #False,
            gradient_type                = "standard", #"component_approx",
            break_condition              = 'percent',
            verbose=True)

    fig, ax = plt.figure(figsize=(15,4)), plt.gca()
    plot_two_comp(ax, vbobj.comp_list, comp_two_text="optimized new comp", scale=.55)
    plt.savefig(os.path.join(OUTPUT_DIR, "mode_two_final.pdf"), bbox_inches='tight')
