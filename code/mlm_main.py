"""
Script that runs the `frisk` and `baseball` models.  Saves inferences
for separate, model-specific plotting scripts.
"""

#########
# cli   #
#########
import argparse, os
parser = argparse.ArgumentParser(
    description="Run vboost, mcmc, and npvi on two hierarchical models")
parser.add_argument('-model', type=str, default='frisk', help="frisk|baseball")
parser.add_argument('-vboost', action='store_true',
                    help='run vboost and save')
parser.add_argument('-vboost_nsamps',  type=int, default=100,
                    help='number of samples for gradient estimates')
parser.add_argument('-npvi',   action='store_true',
                    help='run nonparametric vi and save')
parser.add_argument('-rank',   type=int, default=0,
                    help='rank for vboost components')
parser.add_argument('-ncomp',  type=int, default=20,
                    help='number of components to use with vboost or NPVI')
parser.add_argument('-mcmc',   action='store_true',
                    help='run mcmc (NUTS) and save')
parser.add_argument('-mcmc_nsamps', type=int, default=5000,
                    help='number of MCMC steps to take')
parser.add_argument('-seed', type=int, default=12, help="random seed...")
args, _ = parser.parse_known_args()


############################
# shared model setup code  #
############################
import pandas as pd
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from vbproj import vi
from vbproj.gsmooth.opt import FilteredOptimization
from vbproj.gsmooth.smoothers import AdamFilter
from vbproj.models import baseball, frisk
from vbproj.vi.vboost.components import LRDComponent
import cPickle as pickle

# create output for fit VBoost model
args.output = args.model + "_output"
if not os.path.exists(args.output):
    os.makedirs(args.output)

# set up model function --- autograd-able
def make_model(model_name):
    if model_name == "baseball":
        # baseball model and data
        lnpdf_named = baseball.lnpdf
        lnpdf_flat  = baseball.lnpdf_flat
        lnpdft      = lambda z, t: np.squeeze(lnpdf_flat(z, t))
        lnpdf       = lambda z: np.squeeze(lnpdf_flat(z, 0))
        D           = baseball.D
        return lnpdf, D, None
    elif model_name == "frisk":
        lnpdf, unpack, D, sdf, pnames = frisk.make_model_funs(precinct_type=1)
        return lnpdf, D, pnames

np.random.seed(args.seed)
lnpdf, D, _ = make_model(args.model)
print \
"""
=========== mlm_main ==============

  model                    : {model}
  posterior dimension (D)  : {D}
  output dir               : {output_dir}

""".format(model=args.model, D=D, output_dir=args.output)
print args


# single component initialization
def mfvi_init():
    mfvi = vi.DiagMvnBBVI(lnpdf, D, lnpdf_is_vectorized=True)
    elbo_grad = grad(mfvi.elbo_mc)
    def mc_grad_fun(lam, t):
        return -1.*elbo_grad(lam, n_samps=1024) #args.vboost_nsamps)

    niter = 1000
    lam = np.random.randn(mfvi.num_variational_params) * .01
    mc_opt = FilteredOptimization(
                  mc_grad_fun,
                  lam.copy(),
                  save_grads = False,
                  grad_filter = AdamFilter(),
                  fun = lambda lam, t: mfvi.elbo_mc(lam, n_samps=1000),
                  callback=mfvi.callback)
    mc_opt.run(num_iters=niter, step_size=.1)
    return mc_opt.params.copy()


###########################
# Variational Boosting    #
###########################

if args.vboost:

    # single component MFVI first
    mfvi_lam  = mfvi_init()
    mfvi_file = os.path.join(args.output, "mfvi.npy")
    np.save(mfvi_file, mfvi_lam)

    # Variational Boosting Object (with initial component
    from vbproj.vi.vboost import mog_bbvi
    comp = LRDComponent(D, rank=0, lam=mfvi_lam)
    vbobj = vi.MixtureVI(lambda z, t: lnpdf(z),
                         D                     = D,
                         comp_list             = [(1., comp)],
                         fix_component_samples = True,
                         break_condition='percent')

    # iteratively add comps
    for k in xrange(10):

        # initialize new comp w/ weighted EM scheme
        (init_prob, init_comp) = \
            vbobj.fit_mvn_comp_iw_em(new_rank = comp.rank,
                                     num_samples=2000,
                                     importance_dist = 't-mixture', #'gauss-mixture',
                                     use_max_sample=False)
        init_prob = np.max([init_prob, .5])

        # fit new component
        vbobj.fit_new_comp(init_comp = init_comp,
                           init_prob = init_prob,
                           max_iter  = 100,
                           step_size = .05,
                           num_new_component_samples   =10*D,
                           num_previous_mixture_samples=10*D,
                           fix_component_samples=True,
                           gradient_type="standard", #component_approx_static_rho",
                           break_condition='percent')

    # after all components are added, tune the weights of each comp
    comp_list = mog_bbvi.fit_mixture_weights(vbobj.comp_list, vbobj.lnpdf,
                                            num_iters=1000, step_size=.25,
                                            num_samps_per_component=10*D,
                                            ax=None)
    vbobj.comp_list = comp_list

    # save output here
    vb_outfile = os.path.join(args.output, "vboost.pkl")
    lam_list = [(p, c.lam) for p, c in vbobj.comp_list]
    with open(vb_outfile, 'wb') as f:
        pickle.dump(lam_list, f)


#############################################
# Nonparametric variational inference code  #
#  --- save posterior parameters            #
#############################################

if args.npvi:

    init_with_mfvi = True
    if init_with_mfvi:
        mfvi_lam = mfvi_init()

        # initialize theta
        theta_mfvi = np.atleast_2d(np.concatenate([ mfvi_lam[:D],
                                                    [2*mfvi_lam[D:].mean()] ]))
        mu0        = vi.bbvi_npvi.mogsamples(args.ncomp, theta_mfvi)

        # create npvi object
        theta0 = np.column_stack([mu0, np.ones(args.ncomp)*theta_mfvi[0,-1]])

    else:
        theta0 = np.column_stack([10 * np.random.randn(args.ncomp, D),
                                  -2 * np.ones(args.ncomp)])

    # create initial theta and sample
    npvi = vi.NPVI(lnpdf, D=D)
    mu, s2, elbo_vals, theta = npvi.run(theta0.copy(), verbose=False)
    print elbo_vals

    # save output here
    npvi_outfile = os.path.join(args.output, "npvi_%d-comp.npy"%args.ncomp)
    np.save(npvi_outfile, theta)


#########################################
# MCMC code --- save posterior samples  #
#########################################

if args.mcmc:

    import sampyl
    if args.model == "baseball":
        nuts = sampyl.NUTS(baseball.lnp,
                   start={'logit_phi'  : np.random.randn(1),
                          'log_kappa'  : np.random.randn(1),
                          'logit_theta': np.random.randn(D - 2) })

        # keep track of number of LL calls
        cum_ll_evals = np.zeros(args.mcmc_nsamps, dtype=np.int)
        def callback(i):
            cum_ll_evals[i] = lnpdf.called
            if i % 500 == 0:
                print "total lnpdf calls", lnpdf.called

        lnpdf.called = 0
        chain = nuts.sample(args.mcmc_nsamps, burn=0, callback=callback)
        lnpdf.called = 0
        # compute log like of each sample
        lls   = np.array([ baseball.lnp(*c) for c in chain ])
        # save chain
        nuts_dict = {'chain': chain, 'lls': lls, 'cum_ll_evals': cum_ll_evals}

    elif args.model == "frisk":
        nuts      = sampyl.NUTS(lnpdf, start={"th": np.random.randn(D)*.1})
        chain     = nuts.sample(args.mcmc_nsamps, burn=0, callback=None)
        lls       = np.array([ lnpdf(*c) for c in chain ])
        nuts_dict = {'chain': chain, 'lls': lls}

    mcmc_file = os.path.join(args.output, 'mcmc.pkl')
    with open(mcmc_file, 'wb') as f:
        pickle.dump(nuts_dict, f)


