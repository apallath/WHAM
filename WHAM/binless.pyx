"""
Implementation of binless WHAM (described in Shirts M., & Chodera J. D. (2008)) using
- Negative log-likelihood maximization, inspired from Zhu, F., & Hummer, G. (2012)
- Self-consistent iteration

# IN PROGRESS!!!
"""
import numpy as np
import WHAM.lib.potentials
import WHAM.lib.timeseries
import pymbar.timeseries

import scipy.optimize
from scipy.special import logsumexp

from tqdm import tqdm

# Automatic differentiation, for log-likelihood maximization
import autograd.numpy as anp
from autograd import value_and_grad

# Cython optimization for self-consistent iteration
cimport numpy as np


cpdef self_consistent_solver(np.ndarray N_i, np.ndarray M_l, np.ndarray W_il, float tol=1e-7, int maxiter=100000, int logevery=0):
    """Computes optimal parameters g_i by solving the coupled WHAM equations self-consistently
    until convergence. Optimized using Cython

    Args:
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        M_l (np.array of shape (M,)): Array of total bin counts for each bin.
        W_il (np.array of shape (S, M)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.
        maxiter (int): Maximum number of iterations to run solver for (default = 100000).
        logevery (int): Interval to log self-consistent solver error.

    Returns:
        (g_i (np.array of shape(S,)), status (bool, solution status))
    """
    cdef int S = len(N_i)
    cdef int M = len(M_l)

    cdef np.ndarray g_i = 1e-16 * np.ones(S, dtype=np.float)

    # Solution obtained with required tolerance level?
    cdef bint status = False

    cdef int iter
    cdef float tol_check
    cdef np.ndarray g_i_mat, N_i_mat, G_l, G_l_mat, g_i_prev, increment

    for iter in range(maxiter):
        # Convert to matrices for efficient vectorized logsumexp
        g_i_mat = np.repeat(g_i[:, np.newaxis], M, axis=1)
        N_i_mat = np.repeat(N_i[:, np.newaxis], M, axis=1)

        G_l = logsumexp(g_i_mat + W_il, b=N_i_mat, axis=0) - np.log(M_l)

        G_l_mat = np.repeat(G_l[np.newaxis, :], S, axis=0)

        g_i_prev = g_i

        g_i = -logsumexp(W_il - G_l_mat, axis=1)
        g_i = g_i - g_i[0]

        # Tolerance checek
        g_i_prev[g_i_prev == 0] = 1e-16  # prevent divide by zero
        increment = (g_i - g_i_prev) / g_i_prev

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


def compute_betaF_profile(x_it, u_i, beta, solver='log-likelihood', **solverkwargs):
    """Computes the binned free energy profile.

    Args:
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        solver (string): Solution technique to use ['log-likelihood', 'self-consistent', default='log-likelihood'].
        **solverkwargs: Arguments for solver

    Returns:
        (x (np.array), betaF (np.array), status): Free energy profile, and solver status.
    """
    S = len(u_i)

    """Debug
    # Compute biased free energy profiles for each window
    betaF_il = np.zeros((S, M))
    for i in range(S):
        p_il = n_il[i, :] / N_i[i]
        betaF_il[i, :] = -np.log(p_il / delta_x_l)
    """

    # Compute weights
    W_il = np.zeros((S, M))
    for i in range(S):
        W_il[i, :] = -beta * u_i[i](x_l)

    # Get optimal g_i's\
    if solver == 'log-likelihood':
        g_i, status = minimize_NLL_solver(N_i, M_l, W_il, **solverkwargs)
    elif solver == 'self-consistent':
        g_i, status = self_consistent_solver(N_i, M_l, W_il, **solverkwargs)
    else:
        raise ValueError("Requested solution technique not a recognized option.")

    # Convert to matrices for efficient vectorized logsumexp
    g_i_mat = np.repeat(g_i[:, np.newaxis], M, axis=1)
    N_i_mat = np.repeat(N_i[:, np.newaxis], M, axis=1)

    # Compute consensus/WHAM free energy profile
    betaF_l = np.log(delta_x_l) + logsumexp(g_i_mat + W_il, b=N_i_mat, axis=0) - np.log(M_l)

    return betaF_l, status


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
        betaF_l_sample, _ = compute_betaF_profile(x_it_bsample, x_l, u_i, beta, solver=solver, scale_stat_ineff=False, **solverkwargs)

        # Append to samples
        betaF_l_all_samples[:, sidx] = betaF_l_sample

    # Calculate mean
    betaF_l = betaF_l_all_samples.mean(axis=1)

    # Calculate bootstrap error
    betaF_l_berr = betaF_l_all_samples.std(axis=1) / np.sqrt(Nboot)

    return betaF_l, betaF_l_berr
