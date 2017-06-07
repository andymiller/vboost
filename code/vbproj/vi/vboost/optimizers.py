"""Some standard gradient-based stochastic optimizers.
These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.
These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays."""
from __future__ import absolute_import

import autograd.numpy as np
from autograd.util import flatten_func
from builtins import range
from scipy.optimize import line_search


def sgd(grad, init_params, subopt=None, callback=None,
        break_cond=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    # dynamic step sizes
    if np.isscalar(step_size):
        step_size = np.ones(num_iters) * step_size
    assert len(step_size) == num_iters, "step schedule needs to match num iter"

    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback: callback(unflatten(x), i, unflatten(g))
        velocity = mass * velocity - (1.0 - mass) * g
        x = x + step_size[i] * velocity

        if subopt is not None:
            x = subopt(x, g, i)

        if break_cond is not None:
            if break_cond(x, i, g):
                break
    return unflatten(x)


def adam(grad, init_params, subopt=None, callback=None, break_cond=None,
         num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    # dynamic step sizes
    if np.isscalar(step_size):
        step_size = np.ones(num_iters) * step_size
    assert len(step_size) == num_iters, "step schedule needs to match num iter"

    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback: callback(unflatten(x), i, unflatten(g))
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size[i]*mhat/(np.sqrt(vhat) + eps)

        # do line search on last 
        if subopt is not None:
            x = subopt(x, g, i)

        if break_cond is not None:
            if break_cond(x, i, g):
                break

    return unflatten(x)


def rmsprop(grad, init_params, callback=None, num_iters=100,
            step_size=0.1, gamma=0.9, eps=10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    avg_sq_grad = np.ones(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback: callback(unflatten(x), i, unflatten(g))
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return unflatten(x)


