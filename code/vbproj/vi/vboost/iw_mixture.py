from sklearn.mixture import GMM
import numpy as np
import numpy.random as npr
from . import mog, components, misc
from . import mixtures as mix
import scipy.misc as scpm


DEBUG_I, DEBUG_J = 0, 1


#####################################################################
# methods for initializing a mixture component using re-weighted EM #
#####################################################################


def fit_new_component(comp_list, lnpdf,
                      df              = 100,
                      num_samples     = 1000,
                      importance_dist ='t-mixture',
                      resample_samples= True,
                      clamp_existing=True,
                      debug_ax=None,
                      iw_debug_ax=None,
                      use_max_sample=False,
                      use_weighted_mean=False):
    """
    Fit a new mixture using the importance weighted EM

    This function defines an importance distribution (based on the existing
    comp_list), and creates an importance weighted sample.
    We then use weighted EM to fit a single new (diagonal) gaussian component
    to the target, lnpdf.

    The importance distribution is defined such that there are no huge
    importance weights, using a two-step process

        - generate an importance sample from q(x; comp_list) and lnpdf:

                X_l ~ q(x; comp_list)
                W_l = lnpdf(x) / q(x; comp_list)

        - take some set of high weights (either proportionally, or using a 
          multinomial resample).  For instance, take the top $D$ weights, 
          WLOG W_1, ..., W_D, and define weights W_0 = 1 - sum(W_1, ..., W_D),
          and importance dist

                q_prop(x) = \sum W_1 N(X_l, Sigma_L) + W_0 q(x; comp_list) 

    The goal here is to take BIG weights W_{big} from the first step, and
    break them into pieces for a more stable procedure.  This process can be
    repeated recursively, but we stop at 2 levels in this implementation.
    """
    ##
    ## 0. Generated a weighted sample (using the reweighting scheme)
    ##
    X, lnW = generate_reweighted_sample(comp_list, lnpdf, num_samples,
                                        importance_dist=importance_dist,
                                        df=df,
                                        debug_ax=iw_debug_ax)

    print "Use max sample? ", use_max_sample
    ##
    ## 1. fit weighted EM to the sample: all components in comp_list
    ##    are fixed; weights, and new component are free to vary
    ##
    lnW = lnW - scpm.logsumexp(lnW)     # normalized weights
    W   = num_samples * np.exp(lnW)     # data-space weights - number of data points

    if use_weighted_mean:
        # try fitting just a weighted mean to the samples, and then estimating 
        # the weight using a resampling step
        #new_mean = np.dot(np.exp(lnW), X)
        approx_mean = components.mean_comp_list(comp_list)
        zi  = np.random.choice(len(lnW), size=len(lnW), p=np.exp(lnW))
        Xir = X[zi,:]
        dists = np.sum((Xir - approx_mean)**2, axis=1)
        new_mean = Xir[np.argmax(dists), :]
        new_stds = np.std(Xir, axis=0)
        new_log_stds = np.log(new_stds)

        # estimate the weight with a new importance sampling step
        Xnew      = np.random.randn(num_samples, X.shape[1]) * new_stds + new_mean
        llXnew    = misc.mvn_diag_logpdf(Xnew, new_mean, new_log_stds)
        llXnew_pi = lnpdf(Xnew, 0)
        q_logprob, q_sample, _, _ = \
            components.make_new_component_mixture_lnpdf(comp_list)
        Xold   = q_sample(num_samples)
        llXold = q_logprob(Xold)
        llXold_pi = lnpdf(Xold, 0)
        ln_new = scpm.logsumexp(llXnew_pi - llXnew)
        ln_old = scpm.logsumexp(llXold_pi - llXold)
        ln_total = scpm.logsumexp([ln_new, ln_old])
        new_prob = np.exp(ln_new - ln_total)

    else:
        rho, lam = new_component_weighted_EM(comp_list, X, W, num_iter=10,
                                             clamp_existing_params=clamp_existing)
        new_mean, new_log_stds = np.split(lam, 2)
        new_prob     = rho
        #new_log_stds = .5*np.log(mod.covars_[-1,:])
        #new_prob     = mod.weights_[-1]

    print "mod found new component weight to be ", new_prob

    ## instead, return location of highest weighted resampled mean, and 
    ## overestimate of variance
    if use_max_sample:
        new_mean = X[W.argmax()]
        emeans, ecovs, _, _, _, epis = \
            components.comp_list_to_matrices(comp_list)
        init_c = mog.mog_covariance(emeans, ecovs, epis)
        new_log_stds = .5*np.log(np.diag(.001*init_c))
        new_log_stds = np.log(1e-4 * np.ones(len(new_mean)))
        new_prob     = np.min([.5, W.max() / np.sum(W)])

    if debug_ax is not None:
        import aip.vboost.plots as pu
        # plot isocontours of the existing approx to see where the 
        # new mean ended up being initialized
        # plot the new mean
        mxy = np.array([[new_mean[DEBUG_I], new_mean[DEBUG_J]]])
        sxy = np.exp(np.array([[new_log_stds[DEBUG_I], new_log_stds[DEBUG_J]]]))
        pu.plot_mean_and_std_2d(mxy, sxy, debug_ax)
        debug_ax.scatter(mxy[0,0], mxy[0,1], s=20, c='red', label='new mean')

    # update component list
    new_params   = np.concatenate([ new_mean, new_log_stds] )
    new_comps    = [c[1] for c in comp_list] + [new_params]
    new_pis      = [c[0]*(1-new_prob) for c in comp_list] + [new_prob]
    comp_list    = zip(new_pis, new_comps)
    return comp_list


def new_component_weighted_EM(comp_list, X, W, num_iter=5,
                              debug_ax=None,
                              clamp_existing_params=True):
    """
    Given a comp_list, add a new component, fit with weighted em
    """
    if True:
        # first component LLs go unchanged
        lnpdf_C, sample_C, lnpdf_Cplus, init_params = \
               components.make_new_component_mixture_lnpdf(comp_list)
        ll_q0 = lnpdf_C(X)
        D = X.shape[1]

        # resample X's
        zs = np.random.choice(len(W), size=len(W), p=W/len(W))
        X  = X[zs,:]

        # initialize the new component --- this mean may be too conservative
        init_mu    = np.mean(X, 0) #np.sum(X*W[:,None], 0) / np.sum(W)
        init_var   = np.var(X, 0)  #
        #init_var   = np.sum((X-init_mu)**2 * W[:,None], 0) / np.sum(W)
        ##init_mu = sample_C(1000).mean(0)
        #init_var = np.var(sample_C(1000), 0)

        init_lnstd = .5*np.log(init_var)
        lam = np.concatenate([init_mu, init_lnstd])
        #lam = comp_list[-1][1]
        #lam[:D] = X[np.argmax(W)]
        rho = .1

        # em iterations
        for i in xrange(num_iter):

            if debug_ax is not None:
                ii, jj = DEBUG_I, DEBUG_J
                debug_ax.scatter(lam[:D][ii], lam[:D][jj], s=100, c='red')
                debug_ax.text(lam[:D][ii], lam[:D][jj], "iter %d"%i, fontsize=14)
                print "iter %d rho = %2.5f"%(i, rho)

            # compute lngamma_old = ln P(Z=0 | X) and lngamma_new
            ll_new   = misc.mvn_diag_logpdf(X, lam[:D], lam[D:])
            lnjoint  = np.column_stack([ ll_q0 + np.log(1.-rho),
                                         ll_new + np.log(rho) ])
            lngammas = lnjoint - scpm.logsumexp(lnjoint, 1, keepdims=True)

            ll_marg = scpm.logsumexp(lnjoint, 1).mean()
            print "iter %d ll = %2.5f"%(i, ll_marg)
            #print "   rho = ", rho, lngammas

            # weighted M step
            ws = np.exp(lngammas) #* W[:,None]
            rho_new = np.sum(ws, 0) + 1.
            rho = (rho_new / np.sum(rho_new))[1]
            #print ws

            new_mu  = np.sum(X * ws[:,1,None], 0) / np.sum(ws[:,1,None])
            new_var = np.sum((X-new_mu)**2 * ws[:,1,None], 0) / np.sum(ws[:,1,None])
            lam = np.concatenate([new_mu, np.log(np.sqrt(new_var))])

        return rho, lam

    # initialize model object
    num_comp = len(comp_list) + 1
    mod      = GMM(n_components=num_comp, covariance_type='diag', n_iter=1)
    mod.fit(X)

    # initialize all of the means, covs, pis
    existing_means, covars, _, _, _, pis = \
        components.comp_list_to_matrices(comp_list)
    existing_covars = np.array([np.diag(c) for c in covars])

    # initialize model
    mod.weights_[:-1] = pis * .5
    mod.weights_[-1]  = .5
    # set the first mean to be the location of the highest weighted w...
    # set covar to be marginal covariance for the whole thing
    mod.means_[-1,:] = X[W.argmax(), :]
    init_c = mog.mog_covariance(existing_means, covars, pis)
    mod.covars_[-1,:] = np.diag(init_c)

    def clamp_params(mod):
        mod.means_[:-1,:]  = existing_means
        mod.covars_[:-1,:] = existing_covars
        #mod.covars_[-1,:]  = np.diag(init_c)   #todo experiment with clamping covariance!

    clamp_params(mod)

    # run importance weighted EM
    prev_ll = -np.inf
    for i in xrange(num_iter):

        if debug_ax is not None:
            ii, jj = DEBUG_I, DEBUG_J

            for mi in xrange(mod.means_.shape[0]):
                debug_ax.scatter(mod.means_[mi,ii], mod.means_[mi,jj], s=100, c='red')
                debug_ax.text(mod.means_[mi,ii], mod.means_[mi,jj], "iter %d"%i, fontsize=14)

        log_likelihoods, responsibilities = mod.score_samples(X)
        current_log_likelihood = log_likelihoods.mean()
        weighted_responsibilities = responsibilities * W[:,None]
        mod._do_mstep(X, weighted_responsibilities, mod.params,
                            mod.min_covar)
        if clamp_existing_params:
            clamp_params(mod)

        if (current_log_likelihood - prev_ll) < (1e-10*np.abs(prev_ll)):
            print "  current ll increase too small (after %d iters) "%i
            break

        print "ll = %2.4f"%current_log_likelihood
        prev_ll = current_log_likelihood

    return mod




def generate_reweighted_sample(comp_list, lnpdf, num_samples,
                               importance_dist='t-mixture',
                               df=1000, debug_ax=None):
    """ create a weighted sample (hopefully w/ stable weights)

        generate initial weighted sample, X, W. Then choose the top
        half percentile of weights to form a new mixture
        model.  Then draw a weighted sample from this mixture and return.
    """
    # get component list + comp_list moments
    means, covars, icovs, chols, _, pis = \
        components.comp_list_to_matrices(comp_list)
    qmean  = mog.mog_mean(means, pis)
    qcovar = mog.mog_covariance(means, covars, pis)

    # create first
    if importance_dist == 't-mixture':
        print "  using mixture of scaled ts with %d dfs"%df
        # step 1, create the existing Mixture of T distributions
        q_sample  = lambda n: mix.mixture_of_ts_samples(n, means, chols,
                                                        pis, df=df)
        ichols    = np.array([np.linalg.inv(c) for c in chols])
        q_logprob = lambda x: mix.mixture_of_ts_logprob(x, means, ichols,
                                                        pis, df=df)
    elif importance_dist == 'gauss-mixture':
        print "  using mixture of gaussians as proposal distribution"
        q_logprob, q_sample, _, _ = \
            components.make_new_component_mixture_lnpdf(comp_list)
    else:
        raise Exception("Only 't-mixture' and 'gauss-mixture' proposals supported")

    # draw sample from fat-tailed mixture, and get weights
    X, lnW = sample_and_log_weights(lnpdf, q_logprob, q_sample,
                                    num_samples=num_samples)
    lnP = lnW - scpm.logsumexp(lnW)
    P   = np.exp(lnP)

    print np.percentile(lnP, [1, 25, 50, 75, 99])

    # if a weight is much bigger than it should be (i.e. 10 times bigger, 10/N)
    # then resample it
    high_idx = np.where(lnP > np.log(10./len(lnP)))[0]
    #high_idx = np.where(lnP > upper_cutoff)[0]
    print "  gen_weighted_sample (starting with %d samples)"%num_samples
    print "      breaking up %d samples, (%2.2f of importance weight)" % \
                (len(high_idx), np.sum(P[high_idx]))

    old_weight = 1. - np.sum(P[high_idx])

    # now create a new mixture with new samples tacked on as means, and existing
    # covariance taked on as covariance
    new_covar = np.eye(means.shape[1]) * np.diag(qcovar)
    re_means  = np.row_stack([means, X[high_idx,:]])
    re_covars = np.vstack([covars, np.tile(new_covar, (len(high_idx), 1, 1))])
    re_icovs  = np.array([np.linalg.inv(c) for c in re_covars])
    re_dets   = np.array([np.linalg.det(c) for c in re_covars])
    re_chols  = np.array([np.linalg.cholesky(c) for c in re_covars])
    re_pis    = np.concatenate([old_weight*pis, P[high_idx]])
    re_pis   /= re_pis.sum()

    # now create mixture with the new loc scales
    new_q_logprob = lambda x: mog.mog_logprob(x, re_means, re_icovs, re_dets, re_pis)
    new_q_sampler = lambda N: mog.mog_samples(N, re_means, re_chols, re_pis)

    # sample and weights
    rX, rlnW = sample_and_log_weights(lnpdf, new_q_logprob, new_q_sampler,
                                      num_samples=num_samples)

    # in theory we could wrap re_means and re_covars into a comp_list,
    # and then call this function again if we want
    #new_comp_list = components.matrices_to_comp_list(re_means, re_covars, re_pis)

    if debug_ax is not None:
        #num_samples=100
        ##df = 100
        #from VarBoost.iw_mixture import sample_and_weights
        #from VarBoost import mixtures as mix
        #debug_ax.scatter(X[:,0], X[:,1], s=W*10, c='blue')
        print " ==== iw_mixture.reweighted_sample idx  %d, %d ===="%(DEBUG_I, DEBUG_J)
        rP = np.exp(rlnW - scpm.logsumexp(rlnW))
        debug_ax.scatter(rX[:,DEBUG_I], rX[:,DEBUG_J], s=rP*10*rX.shape[0], c='green', label="resampled")
        debug_ax.scatter(X[:,DEBUG_I], X[:,DEBUG_J], s=P*10*X.shape[0], c='blue', label="original IW")
        #for xy, lw in zip(X, lnW):
        #    debug_ax.text(xy[0], xy[1], "%2.4f"%np.exp(lw))

    return rX, rlnW


##################################
## Jointly fit components       ##
##################################

def fit_importance_weighted_mixture(lnpdf, q_logprob, q_sample,
                                    num_components=2,
                                    num_iter  = 10,
                                    num_samples_per_iter = 100,
                                    callback=None):
    for i in xrange(num_iter):

        # generate approximation 
        X, lnW = sample_and_log_weights(lnpdf, q_logprob, q_sample,
                                        num_samples=num_samples_per_iter)
        W = np.exp(lnW)

        # fit reweighted approximation
        mod = importance_EM(num_components, X, W, num_iter=50)

        # update q_logprob and q_sample
        q_logprob = mod.score
        q_sample  = mod.sample

        ##### plot #######
        if callback is not None:
            callback(i, q_logprob, q_sample)

    return mod


def sample_and_log_weights(lnpdf, q_logprob, q_sample, num_samples=100):
    """ sample from distribution and produce importance weights """
    X      = q_sample(num_samples)
    llx_q  = q_logprob(X)
    llx_pi = lnpdf(X, 0)
    lnW    = llx_pi - llx_q
    return X, lnW


# weighted-em --- if each datapoint has some weight associated with it,
# we incorporate that into the maximization step
def importance_EM(num_comp, X, W, num_iter=20):
    # init model
    mod = GMM(n_components=num_comp, covariance_type='diag', n_iter=10)
    mod.fit(X)

    # run importance weighted EM
    for i in xrange(num_iter):
        log_likelihoods, responsibilities = mod.score_samples(X)
        current_log_likelihood = log_likelihoods.mean()
        weighted_responsibilities = responsibilities * W[:,None]
        mod._do_mstep(X, weighted_responsibilities, mod.params,
                            mod.min_covar)
    return mod



if __name__=="__main__":

    # test new component weighted EM
    # set up test example
    means  = np.array([[-1, 1], [1, -1]])
    covars = np.array([np.eye(2) for _ in xrange(2)])
    icovs  = np.array([np.linalg.inv(c) for c in covars])
    dets   = np.array([np.linalg.det(c) for c in covars])
    pis    = np.array([.5, .5])
    lnpdf = lambda x, i: mog.mog_logprob(x, means, icovs, dets, pis)

    X = mog.mog_samples(1000, means,
                        np.array([np.linalg.cholesky(c) for c in covars]),
                        pis)
    W = np.ones(X.shape[0])

    lam0 = np.concatenate([ means[0], .5*np.log(np.diag(covars[0])) ])
    comp_list = [(1., lam0)]

    mod = new_component_weighted_EM(comp_list, X, W, num_iter=50)
    print "GT Means        : ", means
    print "Inferred Means  : ", mod.means_
    print "Inferred Covars : ", mod.covars_
    print "Inferred Weights: ", mod.weights_

    print "\n"
    gmod = GMM(n_components=2)
    gmod.fit(X)
    print "full em means   : ", gmod.means_
    print "full em weights : ", gmod.weights_
    print "full em covars  : ", gmod.covars_


    # test higher level method
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns
    import autil.util.plots as pu
    fig, ax = plt.figure(figsize=(8,8)), plt.gca()
    pu.plot_isocontours(ax, lambda x: np.exp(lnpdf(x,0)), fill=True)
    new_comp_list = fit_new_component(comp_list, lnpdf, df=10000,
                                      num_samples=1000,
                                      importance_dist='tmixture',
                                      resample_samples=True,
                                      debug_ax=ax)
    ax.legend(loc='best')
    print new_comp_list

    # look at the true ratio everywhere
    fig, ax = plt.figure(figsize=(8,8)), plt.gca()
    single_dist = mog.MixtureOfGaussians(means[:1,:], covars[:1,:,:], np.array([1.]))
    lnratio = lambda x: np.exp(lnpdf(x, 0) - single_dist.logpdf(x))
    cim = pu.plot_isocontours(ax, lnratio, fill=True)
    fig.colorbar(cim)

