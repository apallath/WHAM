"""Implementation of binned WHAM using
- Negative log-likelihood maximization as described in Zhu, F., & Hummer, G. (2012)
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


def NLL(g_i, N_i, M_l, W_il):
    """Computes the negative log-likelihood objective function to minimize.

    Args:
        g_i (np.array of shape (S,)) Array of total free energies associated with the windows
            0, 1, 2, ..., S-1.
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        M_l (np.array of shape (M,)): Array of total bin counts for each bin.
        W_il (np.array of shape (S, M)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.

    Returns:
        A(g) (np.float): Negative log-likelihood objective function.
    """
    term1 = -anp.sum(N_i * g_i)
    log_p_l = anp.log(M_l) - numeric.alogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0)
    term2 = -anp.sum(M_l * log_p_l)

    A = term1 + term2

    return A


def minimize_NLL_solver(N_i, M_l, W_il, g_i=None, opt_method='BFGS', debug=False):
    """Computes optimal g_i by minimizing the negative log-likelihood
    for jointly observing the bin counts in the indepedent windows in the dataset.

    Any optimization method supported by scipy.optimize can be used. BFGS is used
    by default. Gradient information required for BFGS is computed using autograd.

    Args:
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        M_l (np.array of shape (M,)): Array of total bin counts for each bin.
        W_il (np.array of shape (S, M)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.
        g_i (np.array of shape (S,)): Total free energy initial guess.
        opt_method (string): Optimization algorithm to use (default: BFGS).
        debug (bool): Display optimization algorithm progress (default=False).

    Returns:
        (
            g_i (np.array of shape(S,)),
            status (bool): Solution status.
        )
    """
    if g_i is None:
        g_i = np.random.rand(len(N_i))  # TODO: Smarter initial guess

    # Optimize
    res = scipy.optimize.minimize(value_and_grad(NLL), g_i, jac=True,
                                  args=(N_i, M_l, W_il),
                                  method=opt_method, options={'disp': debug})

    g_i = res.x
    g_i = g_i - g_i[0]

    return g_i, res.success


cpdef self_consistent_solver(np.ndarray N_i, np.ndarray M_l, np.ndarray W_il,
                             np.ndarray g_i=np.zeros(1), float tol=1e-7, int maxiter=100000, int logevery=0):
    """Computes optimal parameters g_i by solving the coupled WHAM equations self-consistently
    until convergence. Optimized using Cython.

    Args:
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        M_l (np.array of shape (M,)): Array of total bin counts for each bin.
        W_il (np.array of shape (S, M)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.
        g_i (np.array of shape (S,)): Total free energy initial guess.
        tol (float): Relative tolerance to stop solver iterations at (defaul=1e-7).
        maxiter (int): Maximum number of iterations to run solver for (default=100000).
        logevery (int): Interval to log self-consistent solver error (default=0, i.e. no logging).

    Returns:
        (
            g_i (np.array of shape(S,)),
            status (bool): Solution status.
        )
    """
    cdef float EPS = 1e-24
    cdef int S = len(N_i)
    cdef int M = len(M_l)

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
        G_l = numeric.clogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0) - np.log(M_l)

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
            if logevery is not None:
                print("Self-consistent solver error = {:.2e}.".format(tol_check))
            status = True
            break

    return g_i, status


def compute_betaF_profile(x_it, x_l, u_i, beta, bin_style='left', solver='log-likelihood', scale_stat_ineff=False, **solverkwargs):
    """Computes the binned free energy profile.

    Args:
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        x_l (np.array): Array of bin left-edges/centers of length M.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.
        solver (string): Solution technique to use ['log-likelihood', 'self-consistent', 'debug', default='log-likelihood'].
        scale_stat_ineff (boolean): Compute and scale bin count by statistical inefficiency (default=False).
        **solverkwargs: Arguments for solver

    Returns:
        (
            betaF_l (np.array): Array of WHAM/consensus free energies of length M,
            betaF_il (np.array): Array of biased free energies for each window,
            g_i (np.array): Total window free energies,
            status (bool): Solver status.
        )
    """
    S = len(u_i)
    M = len(x_l)

    # Compute bin edges
    bin_edges = np.zeros(M + 1)

    if bin_style == 'left':
        bin_edges[:-1] = x_l
        # add an artificial edge after the last center
        bin_edges[-1] = x_l[-1] + (x_l[-1] - x_l[-2])
    elif bin_style == 'center':
        # add an artificial edge before the first center
        bin_edges[0] = x_l[0] - (x_l[1] - x_l[0]) / 2
        # add an artificial edge after the last center
        bin_edges[-1] = x_l[-1] + (x_l[-1] - x_l[-2]) / 2
        # bin edges are at the midpoints of bin centers
        bin_edges[1:-1] = (x_l[1:] + x_l[:-1]) / 2
    else:
        raise ValueError('Bin style not recognized.')

    delta_x_l = bin_edges[1:] - bin_edges[:-1]

    # Compute statistical inefficiencies if required, else set as 1 (=> no scaling)
    # For now, use pymbar instead of WHAM.lib.timeseries function
    stat_ineff = np.ones(S)
    if scale_stat_ineff:
        for i in range(S):
            stat_ineff[i] = pymbar.timeseries.statisticalInefficiency(x_it[i], fft=True)

    # Bin x_it
    n_il = np.zeros((S, M))
    for i in range(S):
        n_il[i, :], _ = np.histogram(x_it[i], bins=bin_edges)
        n_il[i, :] = n_il[i, :] / stat_ineff[i]

    N_i = n_il.sum(axis=1)
    M_l = n_il.sum(axis=0)

    # Compute biased free energy profiles for each window
    betaF_il = np.zeros((S, M))
    for i in range(S):
        p_il = n_il[i, :] / N_i[i]
        betaF_il[i, :] = -np.log(p_il / delta_x_l)

    # Compute weights
    W_il = np.zeros((S, M))
    for i in range(S):
        W_il[i, :] = -beta * u_i[i](x_l)

    # Get optimal g_i's\
    if solver == 'log-likelihood':
        g_i, status = minimize_NLL_solver(N_i, M_l, W_il, **solverkwargs)
    elif solver == 'self-consistent':
        g_i, status = self_consistent_solver(N_i, M_l, W_il, **solverkwargs)
    elif solver == 'debug':
        g_i = np.array(solverkwargs.get('g_i'))
        status = True
    else:
        raise ValueError("Requested solution technique not a recognized option.")

    # Compute consensus/WHAM free energy profile
    betaF_l = np.log(delta_x_l) + numeric.clogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0) - np.log(M_l)

    return betaF_l, betaF_il, g_i, status


def bootstrap_betaF_profile(Nboot, x_it, x_l, u_i, beta, solver='log-likelihood', track_progress=False, **solverkwargs):
    """Computes the binned free energy profile with error bars from correlelated timeseries data
    by bootstrapping the data N_boot times."""
    S = len(u_i)
    M = len(x_l)
    betaF_l_all_samples = np.zeros((M, Nboot))

    # Precompute statistical inefficiencies
    g_vec = np.ones(S)
    for i in range(S):
        g_vec[i] = pymbar.timeseries.statisticalInefficiency(x_it[i], fft=True)

    if track_progress:
        Nbootrange = tqdm(range(Nboot))
    else:
        Nbootrange = range(Nboot)
    for sidx in Nbootrange:
        # Draw bootstrap sample
        x_it_bsample = [None] * S
        for i in range(S):
            x_it_bsample[i] = WHAM.lib.timeseries.bootstrap_independent_sample(x_it[i], g_vec[i])

        # Compute free energy
        betaF_l_sample, _, _ = compute_betaF_profile(x_it_bsample, x_l, u_i, beta, solver=solver, scale_stat_ineff=False, **solverkwargs)

        # Append to samples
        betaF_l_all_samples[:, sidx] = betaF_l_sample

    # Calculate mean
    betaF_l = betaF_l_all_samples.mean(axis=1)

    # Calculate bootstrap error
    betaF_l_berr = betaF_l_all_samples.std(axis=1) / np.sqrt(Nboot)

    return betaF_l, betaF_l_berr
