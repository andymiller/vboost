import autograd.numpy as np
from autograd import grad
from .optimizers import sgd, adam
from scipy import optimize
from . import mixtures as mix
from . import normal_bbvi, mog, components, lowr_mvn
from .misc import logit, sigmoid, simplex_to_unconstrained, \
                                  unconstrained_to_simplex
from . import iw_mixture


class MixtureVI:
    """
    Class for boosting a variational inference model

    Inputs
    ------
    lnpdf     : unnormalized distribution, lnpdf(th, i)
    D         : dimensionality of the RV for lnpdf
    comp_list : 

    Optional
    --------
    num_samples      : number of draws for the Monte Carlo estimate
    num_iters        : number of iterations to run
    ...

    """
    def __init__(self, lnpdf, D, comp_list = [], **kwargs):
        self.lnpdf = lnpdf
        self.D     = D
        self.comp_list = comp_list
        self.comp_info = {}

        # debug arguments
        self.debug_i, self.debug_j = 2, 1
        iw_mixture.DEBUG_I, iw_mixture.DEBUG_J = self.debug_i, self.debug_j

    def fit_new_comp(self, init_comp=None, init_prob=None, **kwargs):
        # 0. initialize component if not already passed in (mvn comp here)
        if init_comp is None:
            init_comp, init_prob = self.fit_mvn_comp_iw_em(**self.init_args)

        # 1. fit new component, holding existing comp list fixed
        new_comp_list, new_comp_dict = \
            fit_new_component(self.comp_list, self.lnpdf,
                              init_comp, init_prob,
                              #vb_opt = self.args,
                              **kwargs)

        # 2. organize component list
        self.comp_list = new_comp_list
        self.comp_info[len(self.comp_list)] = new_comp_dict

    def fit_mvn_comp_iw_em(self, new_rank, **kwargs):
        """ function used to initialize a new Multivariate Normal component
        using importance weights and expectation maximization
        """
        iw_comps = \
            iw_mixture.fit_new_component(self.comp_list, self.lnpdf, **kwargs)

        # unpack parameters, create component object
        D = self.comp_list[0][1].D
        init_prob_new, init_lam = iw_comps[-1]
        init_prob_new = np.max([init_prob_new, .01])
        init_mean  = init_lam[:D]
        init_lnstd = init_lam[D:]

        # create component object
        unpacker, setter, n_params = \
            components.make_param_unpacker(D, new_rank)
        comp0 = self.comp_list[0][1]
        m0, C0, v0 = comp0.unpack(comp0.lam)
        init_comp_params = np.zeros(n_params)
        init_comp_params = setter(init_comp_params,
                                  mean = init_mean, #init_lam[:D],
                                  C    = .001*np.random.randn(D, new_rank),
                                  v    = 2*init_lnstd) #init_lam[D:2*D])
        new_comp = components.LRDComponent(D, new_rank, lam=init_comp_params)
        return init_prob_new, new_comp

    def sample(self, num_samps):
        pis = np.array([p for p, _ in self.comp_list])
        indices = np.random.choice(len(self.comp_list), p=pis, size=num_samps)
        samps = np.array([ self.comp_list[i][1].sample(num_samps=1).squeeze()
                           for i in indices ])
        return samps

    def elbo_mc(self, comp_list=None, n_samps=1000):
        qlnpdf, qsample = components.comp_list_to_lnpdf(comp_list)
        z    = qsample(n_samps)
        lnqs = qlnpdf(z)
        lls  = self.lnpdf(z)
        return np.mean(lls - lnqs)


#############################################
# Main fitting method in this module        #
#############################################

def fit_new_component(comp_list,
                      lnpdf,
                      init_comp,
                      init_prob,
                      debug_ax=None,
                      seed=None,
                      ax=None, xlim=None, ylim=None,
                      elbo_ax=None,
                      **kwargs):
    """
    main fitting method for Variational Boosting: 
    given a list of fixed mixture components, fit a new component
    to obtain a tighter lower bound on the log marg like 

    Two sets of arguments --- one controls the initialization of the 
    new component parameters.  the other controsl the optimization of the
    new component
    """
    # unpack vboost params
    max_iter                     = kwargs.get("max_iter", 1000)
    step_size                    = kwargs.get("step_size", .1)
    num_previous_mixture_samples = kwargs.get("num_previous_mixture_samples", 200)
    num_new_component_samples    = kwargs.get("num_new_component_samples", 10)
    fix_component_samples        = kwargs.get("fix_component_samples", True)
    fixed_prob_new               = kwargs.get("fixed_prob_new", False)
    gradient_type                = kwargs.get("gradient_type", "component_approx")
    rho_line_search              = kwargs.get("rho_line_search", False)
    break_condition              = kwargs.get("break_condition", "percent")
    use_adam                     = kwargs.get("use_adam", True)

    print """fitting new component with params:
     step_size : {ss}
     max_iter  : {mi}
     fix_component_samples : {fcs}
    """.format(ss=step_size, mi=max_iter, fcs=fix_component_samples)

    print " adding component --- using %d new samples "%num_new_component_samples

    #########################################################
    # create objective and objective gradient               #
    #########################################################
    new_rank = init_comp.rank
    mixture_elbo = \
        make_mixture_elbo(lnpdf,
              comp_list, new_rank,
              num_previous_mixture_samples=num_previous_mixture_samples,
              num_new_component_samples=num_new_component_samples,
              fix_samples=fix_component_samples)

    # define objective --- add RHO penalty for stability
    def mixture_obj(x, t):
        return -.1*mixture_elbo(x, t) #+ (1/100.)*(sigmoid(x[-1])**2)

    # create flat initial parameter vector
    init_params = np.concatenate([init_comp.lam, [logit(init_prob)]])

    # fix rho if called for
    if fixed_prob_new:
        fixed_prob      = 1. / (len(comp_list) + 1.)
        init_params[-1] = logit(fixed_prob)
        def mixture_obj(x, t):
            return -.1*mixture_elbo(x, t, fixed_prob_new=fixed_prob_new)
        #init_params  = np.concatenate([init_mean, init_log_std,
        #                               [logit(fixed_prob_new)]])

    #print "Params going in: ", init_params.shape
    print "Made objective --- value is ", mixture_obj(init_params, 0)
    print "  made objective gradient is", grad(mixture_obj)(init_params, 0)

    no_new_params = init_params.copy()
    no_new_params[-1] = -20
    print "  made objective with no new comp is", mixture_elbo(no_new_params, 0)

    ######################################################################
    # choose the gradient that we'll be using, set up optimization func  #
    ######################################################################
    mixture_obj_grad = make_gradient_function(mixture_obj, gradient_type,
                                          init_comp.unpack, init_comp.setter)

    # set up line search
    if rho_line_search:
        ls_dir = np.zeros(len(init_params)); ls_dir[-1] = 1.
        def ls_subopt(x, g, i):
            lam_mag = np.sqrt(np.mean(g[:-1]**2))
            rho_mag = np.sqrt(g[-1]**2)
            if rho_mag > 10*lam_mag:
                print " ----- VBoost doing rho Line Search ------ "
                ff = lambda x: mixture_obj(x, i)
                gg = lambda x: mixture_obj_grad(x, i)
                alpha0, fc, gc, _, _, _ = \
                    optimize.line_search(ff, gg, xk=x, pk=ls_dir)
                if alpha0 is not None:
                    print "new rho = ", (x + alpha0*ls_dir)[-1]
                    x = x + alpha0*ls_dir
            return x
    else:
        ls_subopt = None

    # set up the optimization proceudre 
    if use_adam:
        print " === optimizing with ADAM ==== "
        opt_fun = lambda cb, bc: adam(mixture_obj_grad, init_params,
                                      subopt    = ls_subopt,
                                      step_size = step_size,
                                      num_iters = max_iter,
                                      callback=cb, break_cond=bc)

    else:
        opt_fun = lambda cb, bc: sgd(mixture_obj_grad, init_params,
                                     subopt    = ls_subopt,
                                     step_size = step_size,
                                     num_iters = max_iter,
                                     mass      = .001,
                                     callback  = cb, break_cond = bc)

    # create the callback
    param_vals = []
    elbo_vals  = []
    #callback = make_vboost_callback(elbo_vals, param_vals)
    def callback(params, i, g):
        elbo_val = mixture_elbo(params, i)
        elbo_vals.append(elbo_val)
        param_vals.append(params)
        if i % 20 == 0:
            print "iter %d, lower bound %2.4f"%(i, elbo_val)
            # divide up gradient into mean, covariance, diag and rho
            glam, grho = g[:-1], g[-1]
            gm, gC, gv = init_comp.unpack(glam)
            print "  prob new = %2.5f"%sigmoid(params[-1])
            print "  gradient magnitudes "
            print "     gmu = %2.5f " % np.sqrt(np.mean(gm**2))
            print "     gC  = %2.5f " % np.sqrt(np.mean(gC**2))
            print "     gv  = %2.5f " % np.sqrt(np.mean(gv**2))
            print "     rho = %2.5f " % grho

            if ax is not None:
                import matplotlib.pyplot as plt; plt.ion()
                import seaborn as sns; sns.set_style('white')
                import autil.util.plots as pu
                ax.cla()
                # background isocontours (target) + foreground isocontours (approx)
                pu.plot_isocontours(ax, lambda x: np.exp(lnpdf(x, i)),
                                    xlim=xlim, ylim=ylim, fill=True)
                pu.plot_isocontours(ax, lambda x: np.exp(lnpdf_Cplus(x, params)),
                                    xlim=xlim, ylim=ylim, colors='darkred')
                plt.draw()
                plt.pause(1./30.)

        if i % 10 == 0:
            if elbo_ax is not None:
                import matplotlib.pyplot as plt; plt.ion()
                import seaborn as sns; sns.set_style('white')
                import autil.util.plots as pu
                elbo_ax.cla()

                # look at mixture_elbo with rho on the x axis, and gradient on y axis
                lam_dir = g[:-1] / np.sqrt(np.sum(g[:-1]**2))
                curr_lam = params[:-1]
                def pfun(th):
                    z, rho = th
                    lam = np.concatenate([ curr_lam + lam_dir * z, [rho] ])
                    return mixture_elbo(lam, i)

                def rfun(rho):
                    lam = np.concatenate([ curr_lam, [rho] ])
                    return mixture_elbo(lam, i)

                curr_rho = params[-1]
                rho_grid = np.linspace(curr_rho - 6, curr_rho + 2, 100)
                egrid    = np.array([rfun(rho) for rho in rho_grid])
                elbo_ax.plot(rho_grid, egrid)
                elbo_ax.scatter(curr_rho, rfun(curr_rho), c='red', label='curr rho')
                elbo_ax.scatter(rho_grid[egrid.argmax()], egrid.max(), c='blue', label='best rho')
                elbo_ax.legend()
                #co = pu.plot_isocontours(elbo_ax, pfun, xlim=[-5, 5],
                #                                   ylim=[curr_rho - 10, curr_rho+2],
                #                                   vectorized=False,
                #                                   numticks=30, levels=np.linspace(-80, -20, 10))
                #elbo_ax.clabel(co, inline=1, fontsize=12)
                #elbo_ax.scatter(0, curr_rho)
                #elbo_ax.set_xlabel("lambda_c")
                #elbo_ax.set_ylabel("rho")
                plt.draw()
                plt.pause(1.)

    if break_condition=="percent":
        def break_cond(x, i, g):
            elbo_diff = elbo_vals[i-1] - elbo_vals[i]
            percent_change = np.abs(elbo_diff / elbo_vals[i-1])
            if percent_change < 1e-8 and i > 25:
                print "Breaking at iter %d --- elbo vals went from "%i,
                print "   %2.4f to %2.4f (%2.4f percent change)" % \
                      (elbo_vals[i-1], elbo_vals[i], percent_change)
                return True
            return False
    elif break_condition=="gradmag":
        def break_cond(x, i, g):
            gmag = np.sqrt(np.sum(g**2))
            if gmag < 1e-5 and i > 100:
                return True
            return False
    else:
        break_cond = lambda x, i, g: False

    # run optimization, keep track of elbo values along the way
    new_comp_params = opt_fun(callback, break_cond)
    elbo_vals       = np.array(elbo_vals)

    # update component list
    # unpack new var params --- compute normalized rho's
    lam_Cplus, rho = new_comp_params[:-1], new_comp_params[-1]

    # compute new pis
    prob_new = sigmoid(rho)
    pis_old = np.array([c[0] for c in comp_list])
    pis_new = np.concatenate([pis_old * (1-prob_new), [prob_new]])
    print "new prob for new comp: %2.5f"%prob_new

    # update new comp_list
    # create new component out of lam
    comp = components.LRDComponent(init_comp.D, new_rank, lam=lam_Cplus)
    comps_new = [c[1] for c in comp_list] + [comp] #[lam_Cplus]
    comp_list_new = zip(pis_new, comps_new)
    return comp_list_new, (elbo_vals, None, np.array(param_vals))


def make_mixture_elbo(lntarget, comp_list, new_rank,
                      num_previous_mixture_samples = 200,
                      num_new_component_samples    = 10,
                      fix_samples                  = True):
    """
    Args:
        lntarget    : log probability function we are approximating
        lnpdf_C     : previous approximation log probability function  (with
                      C fixed components)
        sample_C    : previous approximation sampling function
        lnpdf_Cplus : new approximation sampling function (with C fixed
                      components and 1 varying component + weight)

    Returns:
        mixture_elbo: function that takes in a param vector th = [lam_new, rho]
                      and returns a Monte Carlo approx of the evidence lower
                      bound for lntarget approximated by lnpdf_Cplus(x; th)
    """
    # create the C component variational distributions, sample from it
    lnpdf_C, sample_C, lnpdf_Cplus, (unpacker, setter, n_params) = \
        components.make_new_component_mixture_lnpdf(comp_list, new_rank)
    C_samps    = sample_C(num_previous_mixture_samples)
    llC        = lnpdf_C(C_samps)
    lntarget_C = lntarget(C_samps, 0)
    print llC

    # create the new C+1 component variational dist as a function of lam, rho
    lnpdf_Cplus_fixed_samps = lambda x, th: lnpdf_Cplus(x, th, llC=llC)

    # elbo oparams
    Nsamps, D  = C_samps.shape
    Neps       = num_new_component_samples

    def mixture_elbo(th, i, eps_lowr=None, eps_diag=None,
                            fixed_prob_new=None):
        """
        Computes the approximate lower bound on the log marginal
        likelihood.  th = [component_params, logit(pnew)]
        """
        # unpack and compute prev/new mixing weights
        lam_new, rho_new = th[:-1], th[-1]
        if fixed_prob_new is not None:
            prob_new = fixed_prob_new
            th = np.concatenate([lam_new, [logit(prob_new)]])
            print "fixed prob new"
        else:
            prob_new = sigmoid(rho_new)

        # compute previous component term
        #Cterm = np.mean(lntarget_C - lnpdf_Cplus_fixed_samps(C_samps, th))
        Cterm = np.mean(lntarget_C - lnpdf_Cplus(C_samps, th, llC=llC))

        # compute new component term using reparam trick
        if eps_lowr is None:
            eps_lowr = np.random.randn(Neps, new_rank)
        if eps_diag is None:
            eps_diag = np.random.randn(Neps, D)

        if new_rank == 0:
            mean, log_std = lam_new[:D], lam_new[D:]
            samples = eps_diag * np.exp(.5*log_std) + mean
        else:
            m, C, v = unpacker(lam_new)
            samples = lowr_mvn.mvn_lowrank_sample(Neps, m, C, v,
                                                  eps_lowr=eps_lowr,
                                                  eps_diag=eps_diag)

        lnpi   = lntarget(samples, i)
        lnCp   = lnpdf_Cplus(samples, th)
        Cpterm = np.mean(lntarget(samples, i) - lnpdf_Cplus(samples, th))

        # compute mixing
        return (1. - prob_new) * Cterm + prob_new * Cpterm


    if fix_samples:
        print "Fixing eps samples"
        print "    D, r = ", D, new_rank
        np.random.seed(42)
        Neps = num_new_component_samples
        eps_diag = np.random.randn(Neps, D)
        eps_lowr = np.random.randn(Neps, new_rank)
        return lambda th, i, fixed_prob_new=None: mixture_elbo(th, i, eps_lowr=eps_lowr,
                                                 eps_diag=eps_diag, fixed_prob_new=fixed_prob_new)

    return mixture_elbo


def make_gradient_function(mixture_obj, gradient_type, unpacker, setter):
    """ chooses gradient among different approximations """
    if gradient_type == "standard":
        mixture_obj_grad = grad(mixture_obj)

    elif gradient_type == "score_mean":
        print "Scoring the mean with the current variance!!!"
        # scores the component of the parameter that make the mean with
        # the covariance
        def mixture_obj_grad(x, t):
            # build the covariance
            m, C, v = unpacker(x)
            Sig = np.dot(C, C.T) + np.diag(np.exp(v))

            # grab the mean component of the gradient
            g = grad(mixture_obj)(x, t)
            mg, Cg, vg = unpacker(g)

            # score the mean component of the gradient
            g = setter(g, mean=np.dot(Sig, mg))
            return g

    elif gradient_type == "normal_approx_natural":
        raise NotImplementedError()
        # TODO unpack component params --- mog means, covs, pis
        # TODO move this to a generic 
        # unpack component params
        def mixture_fisher(x):
            mu  = mog_mean(x)
            cov = mog_cov(x)
            Jmu  = jacobian(mu)(x)
            Jcov = jacobian(cov)(x)

            cinv = np.linalg.solve(cov)
            fisher = np.dot(np.dot(Jmu.T, cinv), Jmu) + \
                     np.dot(np.dot(cinv, Jcov), np.dot(cinv, Jcov))
            return fisher

        # unpack component params
        def mixture_obj_grad(x, t):
            Fdiag = mixture_fisher(x)
            return grad(mixture_obj)(x, t) * (1./Fdiag)

    elif gradient_type=="component_approx_static_rho":
        print "Using component_approx_static_rho for natural gradient"
        def mixture_obj_grad(x, t):
            # old way need to tune lambda
            lam_fisher = normal_bbvi.fisher_info(x[:-1])
            Fdiag      = np.concatenate([lam_fisher, [1000]])
            return grad(mixture_obj)(x,t) * (1./Fdiag)

    elif gradient_type == "component_approx":
        print "Using component_approx for natural gradient"
        def mixture_obj_grad(x, t):
            # first scale by fisher info
            prob_new   = sigmoid(x[-1])
            dprob_drho = prob_new * (1 - prob_new)
            lam_fisher = normal_bbvi.fisher_info(x[:-1])
            Fdiag = np.concatenate([ (lam_fisher*prob_new),
                                     [(dprob_drho*(1/prob_new))] ])
            #print "obj grad input shape", len(x)
            return grad(mixture_obj)(x,t) * (1./Fdiag)

    else:
        raise NotImplementedError("not one of the available gradient methods")
    return mixture_obj_grad


def make_vboost_callback(elbo_vals, param_vals):
    # first fit a single gaussian using BBVIA
    pass


######################################################################
# fixed means/variances --- optimize the weights of the mixture      #
######################################################################

def fit_mixture_weights(comp_list, lnpdf, num_iters=1000, step_size=.2,
                        num_samps_per_component=100,
                        fix_samples=False,
                        ax=None, xlim=None, ylim=None):

    # define the mixture elbo as a function of only mixing weights. 
    # to do this, we take L samples from each component, and note that
    # the ELBO decomposes into the sum of expectations wrt each component
    #   ELBO(rho) = Eq[lnpi(x) - ln q(x; rho)]
    #             = sum_c rho_c \int q_c(x; rho) [lnpi(x) - ln q(x; rho)]
    #
    # unpack means/covars/pis for existing approx
    L = num_samps_per_component
    means, covars, icovs, chols, lndets, pis = \
        components.comp_list_to_matrices(comp_list)
    C, D = means.shape

    # create qlogprob as a function of inputs and weights
    qlogprob = lambda x, pis: mog.mog_logprob(x, means, icovs, lndets, pis)

    # sample from each component
    Csamps = np.array([ c.sample(L) for _, c in comp_list ])

    # cache values of pi for each cluster sample
    pi_lls      = np.array([ lnpdf(c, 0) for c in Csamps ])
    pi_lls_mean = np.mean(pi_lls, 1)

    # create approximate elbo with samps and lnprob functions
    def weight_elbo(rhos, i):
        # compute mixing vector
        pis = unconstrained_to_simplex(rhos)

        # sample from comps TODO - decide how the MC gradients should be computed
        #eps_tens = np.random.randn(C, L, D)
        #sd_diag  = np.array([np.sqrt(np.diag(c)) for c in covars])
        #Csamps   = eps_tens * sd_diag[:,None,:] + means[:,None,:]

        # compute entropy estimate terms
        lnq_terms = np.reshape(qlogprob(np.reshape(Csamps, (-1, D)), pis),
                               (C, L))
        lnq_means = np.mean(lnq_terms, 1)
        return np.sum(pis * (pi_lls_mean - lnq_means))

    # first fit a single gaussian using BBVI
    def callback(params, i, g):
        if i % 50 == 0:
            print "weight opt iter %d, lower bound %2.4f"%(i, weight_elbo(params, i))
            print "  weights    = ", unconstrained_to_simplex(params)
            print "  gmag, grad = ", np.sqrt(np.sum(g**2)), g

            if ax is not None:
                import matplotlib.pyplot as plt; plt.ion()
                import seaborn as sns; sns.set_style('white')
                import autil.util.plots as pu
                ax.cla()
                # background isocontours (target) + foreground isocontours (approx)
                pu.plot_isocontours(ax, lambda x: np.exp(lnpdf(x, i)),
                                    xlim=xlim, ylim=ylim, fill=True)
                pis = unconstrained_to_simplex(params)
                pu.plot_isocontours(ax, lambda x: np.exp(qlogprob(x, pis)),
                                    xlim=xlim, ylim=ylim, colors='darkred')
                plt.draw()
                plt.pause(1./30.)

    def break_cond(x, i, g):
        gmag = np.sqrt(np.sum(g**2))
        #if gmag < 1e-4:
        #    return True
        return False

    # TODO: figure out non-sgd opt???
    # optimize component
    var_obj      = lambda x, t: -1.*weight_elbo(x, t)
    var_obj_grad = grad(var_obj)
    init_rhos    = simplex_to_unconstrained(pis) #.1*np.random.randn(len(pis)-1)
    fit_rhos     = adam(var_obj_grad, init_rhos, num_iters=num_iters,
                        step_size=step_size, callback=callback,
                        break_cond=break_cond)

    # unpack new var params --- compute normalized rho's
    pis_new       = unconstrained_to_simplex(fit_rhos)
    comps_new     = [c[1] for c in comp_list]
    comp_list_new = zip(pis_new, comps_new)
    return comp_list_new


######################################################################
# fixed means/variances --- optimize the weights of the mixture      #
######################################################################

def fit_mixture_jointly(num_comps, lnpdf, D, num_iters=1000, step_size=.2,
                        num_samps_per_component=100,
                        fix_samples=True,
                        init_comp_list = None,
                        ax=None, xlim=None, ylim=None):

    # define the mixture elbo as a function of only mixing weights. 
    # to do this, we take L samples from each component, and note that
    # the ELBO decomposes into the sum of expectations wrt each component
    #   ELBO(rho) = Eq[lnpi(x) - ln q(x; rho)]
    #             = sum_c rho_c \int q_c(x; rho) [lnpi(x) - ln q(x; rho)]
    C = num_comps
    L = num_samps_per_component

    from autil.util.misc import WeightsParser
    parser = WeightsParser()
    parser.add_shape("ln_weights", (C-1,))
    parser.add_shape("means", (C, D))
    parser.add_shape("lnstds", (C, D))

    init_rhos    = simplex_to_unconstrained(np.ones(C) * (1./C))
    init_vars    = -2 * np.ones((C, D))
    init_means   = .001 * np.random.randn(C, D)
    if init_comp_list is not None:
        assert len(init_comp_list) == C
        pis = np.array([c[0] for c in init_comp_list])
        init_rhos  = simplex_to_unconstrained(pis)
        init_means = np.row_stack([c[1][:D] for c in init_comp_list])
        init_vars  = np.row_stack([c[1][D:] for c in init_comp_list])
    init_params = np.zeros(parser.num_weights)
    init_params = parser.set(init_params, "ln_weights", init_rhos)
    init_params = parser.set(init_params, "means", init_means)
    init_params = parser.set(init_params, "lnstds", init_vars)

    def joint_elbo(params, i, eps_tens=None):
        # sample from each cluster's normal --- transform into 
        if eps_tens is None:
            eps_tens = np.random.randn(C, L, D)
        lnstds   = parser.get(params, "lnstds")
        means    = parser.get(params, "means")
        Csamps   = eps_tens * np.exp(lnstds)[:,None,:] + means[:,None,:]

        # make qln pdf for params
        icovs = np.array([ np.diag(np.exp(-2*lnstds[c]))
                           for c in xrange(lnstds.shape[0]) ])
        dets  = np.exp(np.sum(2*lnstds, 1))
        lnws  = parser.get(params, "ln_weights")
        pis   = unconstrained_to_simplex(lnws)
        qlogprob = lambda x: mog.mog_logprob(x, means, icovs, dets, pis)

        # compute E_q_c[ lnq(x) ] for each component
        lnq_terms = np.reshape(qlogprob(np.reshape(Csamps, (-1, D))), (C, L))
        lnq_means = np.mean(lnq_terms, 1)

        # compute E[pi(x)] for each component
        pi_lls      = np.array([ lnpdf(c, 0) for c in Csamps ])
        pi_lls_mean = np.mean(pi_lls, 1)
        return np.sum(pis * (pi_lls_mean - lnq_means))

    # first fit a single gaussian using BBVI
    def callback(params, i, g):
        if i % 2 == 0:
            print "weight opt iter %d, lower bound %2.4f"%(i, joint_elbo(params, i))
            #print "  weights    = ", unconstrained_to_simplex(params)
            #print "  gmag, grad = ", np.sqrt(np.sum(g**2)), g

            if ax is not None:
                import matplotlib.pyplot as plt; plt.ion()
                import seaborn as sns; sns.set_style('white')
                import autil.util.plots as pu
                ax.cla()
                # background isocontours (target) + foreground isocontours (approx)
                pu.plot_isocontours(ax, lambda x: np.exp(lnpdf(x, i)),
                                    xlim=xlim, ylim=ylim, fill=True)
                pis = unconstrained_to_simplex(params)
                pu.plot_isocontours(ax, lambda x: np.exp(qlogprob(x, pis)),
                                    xlim=xlim, ylim=ylim, colors='darkred')
                plt.draw()
                plt.pause(1./30.)

    def break_cond(x, i, g):
        gmag = np.sqrt(np.sum(g**2))
        #if gmag < 1e-4:
        #    return True
        return False

    if fix_samples:
        eps_tens = np.random.randn(C, L, D)
        var_obj = lambda x, t: -1.*joint_elbo(x, t, eps_tens=eps_tens)
    else:
        var_obj = lambda x, t: -1.*joint_elbo(x, t)

    # optimize component
    var_obj_grad = grad(var_obj)
    #fit_params = adam(var_obj_grad, init_params, num_iters=num_iters,
    #                  step_size=step_size, callback=callback,
    #                  break_cond=break_cond)
    fit_params = sgd(var_obj_grad, init_params, num_iters=num_iters,
                     step_size=step_size, callback=callback,
                     break_cond=break_cond, mass=.01)

    # unpack new var params --- compute normalized rho's
    pis_new = unconstrained_to_simplex(parser.get(fit_params, "ln_weights"))
    means_new = parser.get(fit_params, "means")
    stds_new  = parser.get(fit_params, "lnstds")
    lams_new  = np.column_stack([means_new, stds_new])
    comp_list_new = [(p, l) for p, l in zip(pis_new, lams_new)]
    return comp_list_new

