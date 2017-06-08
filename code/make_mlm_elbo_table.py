import pandas as pd
from mlm_main import make_model
import autograd.numpy as np
from vbproj import vi
from vbproj.vi.vboost.components import LRDComponent
import cPickle as pickle
import pyprind

Nsamp = 100000

def mfvi_elbo(model):
    """ computes the elbo value for the saved MFVI model """
    lnpdf, D, param_names = make_model(model)
    mfvi = vi.DiagMvnBBVI(lnpdf, D, lnpdf_is_vectorized=True)
    mfvi_lam = np.load(model + "_output/mfvi.npy")
    elbo_val = mfvi.elbo_mc(mfvi_lam, n_samps=Nsamp)
    return elbo_val

def npvi_elbo(model, ncomp):
    """ computes the elbo value for saved NPVI models """
    lnpdf, D, param_names = make_model(model)
    npvi = vi.NPVI(lnpdf, D=D)
    npvi_theta = np.load(model + "_output/npvi_%d-comp.npy"%ncomp)
    print "  npvi n comp: ", npvi_theta.shape[0]
    elbo_val = npvi.mc_elbo(npvi_theta, nsamps = Nsamp)
    return elbo_val

def vboost_elbo(model, ncomp, rank):
    """ computes the elbo value for saved VBoost models """
    if ncomp == 1:
        return mfvi_elbo(model)

    lnpdf, D, param_names = make_model(model)
    vboost = vi.MixtureVI(lnpdf, D=D)
    fname = model + "_output/vboost_%d-comp_%d-rank.pkl"%(ncomp-2, rank)
    with open(fname, 'rb') as f:
        lam_list = pickle.load(f)
        comp_list = [(p, LRDComponent(D, rank, lam=c)) for (p, c) in lam_list]
        print "   ncomp = %d, %d"%(len(comp_list), ncomp)

    elbo_val = vboost.elbo_mc(comp_list, n_samps=Nsamp)
    return elbo_val


#################################################################
# create table of the following shape of ELBO values            #
#                                                               #
#          |  1-comp | 2-comp | 5-comp | 10-comp | 20-comp      #
#  vboost  |         |        |        |         |              #
#  npvi    |         |        |        |         |              #
#                                                               #
#################################################################

model = "frisk"
ncomps = [1, 2, 5, 10, 15, 20]

# compute elbos
vboost_elbos = np.array([ vboost_elbo(model, c, 0) for c in ncomps ])
npvi_elbos   = np.array([ npvi_elbo(model, c)      for c in ncomps ])

elbo_df = pd.DataFrame(np.row_stack([vboost_elbos, npvi_elbos]),
                       index   = ["vboost", "npvi"],
                       columns = ["%d"%c for c in ncomps])


print elbo_df

tstr = elbo_df.to_latex(float_format="%2.3f")
with open(model+"_output/npvi_vboost_table.tex", 'w') as f:
    pickle.dump(tstr, f)



