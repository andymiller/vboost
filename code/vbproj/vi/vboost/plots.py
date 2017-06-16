import matplotlib.pyplot as plt
import autograd.numpy as np
from scipy.stats import norm
from matplotlib.patches import Ellipse


def plot_isocontours(ax, func, xlim=[-3, 3], ylim=[-4, 4], numticks=501,
                     fill=False, vectorized=True, colors=None):
    import numpy as np
    x    = np.linspace(*xlim, num=numticks)
    y    = np.linspace(*ylim, num=numticks)
    X, Y = np.meshgrid(x, y)
    pts  = np.column_stack([X.ravel(), Y.ravel()])
    if vectorized:
        Z = func(pts).reshape(X.shape)
    else:
        Z = np.array([ func(xy) for xy in pts ] ).reshape(X.shape)
    if fill:
        return ax.contourf(X, Y, Z, linewidths=2, colors=colors)
    else:
        return ax.contour(X, Y, Z, linewidths=2, colors=colors)


def plot_normal_marginal(marg_idx, lam, xgrid, ax, **kwargs):
    """ lam = [ mu, ln(std) ] """
    D = len(lam) / 2
    mu, std = lam[marg_idx], np.exp(lam[marg_idx+D])
    pgrid = norm.pdf(xgrid, loc=mu, scale=std)
    ax.plot(xgrid, pgrid, **kwargs)
    return ax


def plot_mean_and_std_2d(means, stds, ax, text_labels=None, **kwargs):
    for i, (mean, std) in enumerate(zip(means, stds)):
        ell = Ellipse(xy=mean, width=4*std[0], height=4*std[1], angle=0.)
        ell.set_alpha(.05)
        ell.set_color('red')
        ax.add_artist(ell)
        ax.scatter(mean[0], mean[1], s=100, c='red')

        if text_labels is not None:
            ax.text(mean[0]+.5*std[0], mean[1]+.5*std[1], text_labels[i])


def plot_covariances(Ca, Cb, ax, only_sds=False, corr_coefs=False):
    """ scatter plot comparison for two covariance mats """
    if corr_coefs:
        sda = np.sqrt(np.diag(Ca))
        sdb = np.sqrt(np.diag(Ca))
        Ca = Ca / np.dot(sda[:,None], sda[None,:])
        Cb = Cb / np.dot(sdb[:,None], sdb[None,:])
    if only_sds:
        avals = np.sqrt(np.diag(Ca))
        bvals = np.sqrt(np.diag(Cb))
    else:
        ac = np.tril(Ca, -1)
        avals = ac[ac!=0].flatten()
        bc = np.tril(Cb, -1)
        bvals = bc[bc!=0].flatten()

    lim = [np.min([avals.min(), bvals.min()]),
           np.max([avals.max(), bvals.max()])]
    ax.scatter(avals, bvals)
    ax.plot(lim, lim, c='grey', alpha=.25)

