"""
Implementation of binless WHAM (described in Shirts M., & Chodera J. D. (2008)) using
- Negative log-likelihood maximization, inspired from Zhu, F., & Hummer, G. (2012)
- Self-consistent iteration.
"""
import numpy as np

import WHAM.lib.potentials
import WHAM.lib.timeseries
from WHAM.lib import numeric

import pymbar.timeseries

import scipy.optimize

from tqdm import tqdm

# Automatic differentiation, for log-likelihood maximization
import autograd.numpy as anp
from autograd import value_and_grad

# Cython optimization for self-consistent iteration
cimport numpy as np


def NLL(g_i, x_l, N_i, W_il):
    """Computes the negative log-likelihood objective function to minimize.

    Args:
        g_i (np.array of shape (S,)) Array of total free energies associated with the windows
            0, 1, 2, ..., S-1.
        x_l (np.array of shape (Ntot,)): Array containing each sample.
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        W_il (np.array of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.

    Returns:
        A(g) (np.float): Negative log-likelihood objective function.
    """
    with anp.errstate(divide='ignore'):
        Ntot = anp.sum(N_i)
        term1 = -anp.sum(N_i / Ntot * g_i)
        term2 = 1 / Ntot * anp.sum(numeric.alogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis] / Ntot, axis=0))

        A = term1 + term2

        return A


def minimize_NLL_solver(x_l, N_i, W_il, g_i=None, opt_method='BFGS', debug=False):
    """Computes optimal g_i by minimizing the negative log-likelihood
    for jointly observing the bin counts in the indepedent windows in the dataset.

    Any optimization method supported by scipy.optimize can be used. BFGS is used
    by default. Gradient information required for BFGS is computed using autograd.

    Args:
        x_l (np.array of shape (Ntot,)): Array containing each sample.
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        W_il (np.array of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.
        g_i (np.array of shape (S,)): Total free energy initial guess.
        opt_method (string): Optimization algorithm to use (default: BFGS).
        debug (bool): Display optimization algorithm progress (default=False).

    Returns:
        (
            bG_l (np.array of shape (Ntot,)): Free energy for each sample point,
            g_i (np.array of shape (S,)): Total free energy for each window,
            status (bool): Solution status.
        )
    """
    if g_i is None:
        g_i = np.random.rand(len(N_i))  # TODO: Smarter initial guess

    # Optimize
    res = scipy.optimize.minimize(value_and_grad(NLL), g_i, jac=True,
                                  args=(x_l, N_i, W_il),
                                  method=opt_method, options={'disp': debug})

    g_i = res.x
    g_i = g_i - g_i[0]

    G_l = numeric.clogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0)

    return G_l, g_i, res.success


cpdef self_consistent_solver(np.ndarray x_l, np.ndarray N_i, np.ndarray W_il,
                             np.ndarray g_i=np.zeros(1), float tol=1e-7, int maxiter=100000, int logevery=0):
    """Computes optimal parameters g_i by solving the coupled MBAR equations self-consistently
    until convergence. Optimized using Cython.

    Args:
        x_l (np.array of shape (Ntot,)): Array containing each sample.
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        W_il (np.array of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.
        g_i (np.array of shape (S,)): Total free energy initial guess.
        tol (float): Relative tolerance to stop solver iterations at (defaul=1e-7).
        maxiter (int): Maximum number of iterations to run solver for (default=100000).
        logevery (int): Interval to log self-consistent solver error (default=0, i.e. no logging).

    Returns:
        (
            bG_l (np.array of shape (Ntot,)): Free energy for each sample point,
            g_i (np.array of shape (S,)): Total free energy for each window,
            status (bool): Solution status.
        )
    """
    cdef float EPS = 1e-24

    cdef int S = len(N_i)
    cdef int Ntot = len(x_l)

    if np.allclose(g_i, np.zeros(1)):
        g_i = EPS * np.ones(S, dtype=np.float)
    else:
        g_i[g_i == 0] = EPS

    # Solution obtained with required tolerance level?
    cdef bint status = False

    cdef int iter
    cdef float tol_check
    cdef np.ndarray g_i_mat, N_i_mat, G_l, G_l_mat, g_i_prev, increment

    for iter in range(maxiter):
        G_l = numeric.clogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0)

        g_i_prev = g_i

        g_i = -numeric.clogsumexp(W_il - G_l[np.newaxis, :], axis=1)
        g_i = g_i - g_i[0]

        # Tolerance check
        g_i[g_i == 0] = EPS  # prevent divide by zero
        g_i_prev[g_i_prev == 0] = EPS  # prevent divide by zero
        increment = np.abs((g_i - g_i_prev) / g_i_prev)

        tol_check = increment[np.argmax(increment)]

        if logevery > 0:
            if iter % logevery == 0:
                print("Self-consistent solver error = {:.2e}.".format(tol_check))

        if increment[np.argmax(increment)] < tol:
            if logevery > 0:
                print("Self-consistent solver error = {:.2e}.".format(tol_check))
            status = True
            break

    return G_l, g_i, status


def compute_betaF_profile(x_it, x_bin, u_i, beta, bin_style='left', solver='log-likelihood', **solverkwargs):
    """Computes the binned free energy profile.

    Args:
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        x_bin (list): Array of bin left-edges/centers of length M. Used only for computing final PMF.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.
        solver (string): Solution technique to use ['log-likelihood', 'self-consistent', default='log-likelihood'].
        **solverkwargs: Arguments for solver.

    Returns:
        (
            betaF_bin (np.array): Array of MBAR/consensus free energies of length M,
            betaF_ibin (np.array): Array of biased free energies for each window,
            g_i (np.array): Total window free energies,
            status (bool): Solver status.
        )
    """
    S = len(u_i)
    M = len(x_bin)

    # Compute bin edges
    bin_edges = np.zeros(M + 1)

    if bin_style == 'left':
        bin_edges[:-1] = x_bin
        # add an artificial edge after the last center
        bin_edges[-1] = x_bin[-1] + (x_bin[-1] - x_bin[-2])
    elif bin_style == 'center':
        # add an artificial edge before the first center
        bin_edges[0] = x_bin[0] - (x_bin[1] - x_bin[0]) / 2
        # add an artificial edge after the last center
        bin_edges[-1] = x_bin[-1] + (x_bin[-1] - x_bin[-2]) / 2
        # bin edges are at the midpoints of bin centers
        bin_edges[1:-1] = (x_bin[1:] + x_bin[:-1]) / 2
    else:
        raise ValueError('Bin style not recognized.')

    delta_x_bin = bin_edges[1:] - bin_edges[:-1]

    # Combine x_it into x_l, where each data point is its own bin
    x_l = x_it[0]
    for i in range(1, S):
        x_l = np.hstack((x_l, x_it[i]))
    Ntot = len(x_l)

    # Compute window counts
    N_i = np.array([len(arr) for arr in x_it])

    # Compute weights for each data point
    W_il = np.zeros((S, Ntot))
    for i in range(S):
        W_il[i, :] = -beta * u_i[i](x_l)

    # Get free energy profile through binless WHAM/MBAR
    if solver == 'log-likelihood':
        G_l, g_i, status = minimize_NLL_solver(x_l, N_i, W_il, **solverkwargs)
    elif solver == 'self-consistent':
        G_l, g_i, status = self_consistent_solver(x_l, N_i, W_il, **solverkwargs)
    else:
        raise ValueError("Requested solution technique not a recognized option.")

    # Construct consensus free energy profiles by constructing weighted histogram
    # using individual point weights G_l obtained from solver
    betaF_bin = np.zeros(M)
    for b in range(1, M + 1):
        sel_mask = np.logical_and(x_l >= bin_edges[b - 1], x_l < bin_edges[b])
        G_l_bin = G_l[sel_mask]
        betaF_bin[b - 1] = np.log(delta_x_bin[b - 1]) - numeric.clogsumexp(-G_l_bin)

    return betaF_bin, g_i, status
