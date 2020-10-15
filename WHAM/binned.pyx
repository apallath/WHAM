"""Implementation of binned WHAM using negative log-likelihood maximization
as described in Zhu, F., & Hummer, G. (2012)."""
import numpy as np
import WHAM.lib.potentials
import WHAM.lib.timeseries
import pymbar.timeseries

import scipy.optimize
from scipy.special import logsumexp

from tqdm import tqdm

cimport numpy as np


def NLL(g_i, N_i, Ntot, M_l, W_il):
    """Computes negative log-likelihood objective function.

    Args:
        g_i (np.array of shape (S-1,)): Array of total free energies associated with the windows
            0, 1, 2, ..., S-2. Free energy of the last window is fixed at 0 (reference value).
        N_i (np.array of shape (S-1,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-2.
        Ntot (float): Sum of sample counts over all windows.
        M_l (np.array of shape (M,)): Array of combined bin counts for each bin.
        W_il (np.array of shape (S-1, M)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-2.

    Returns:
        A(g) (np.float): Negative log-likelihood objective function.
    """
    g_i_mat = np.repeat(g_i[:, np.newaxis], len(M_l), axis=1)
    N_i_mat = np.repeat(N_i[:, np.newaxis], len(M_l), axis=1)

    A = -np.sum(N_i * g_i) - np.sum(M_l * (np.log(M_l) - logsumexp(g_i_mat + W_il, b=N_i_mat, axis=0)))

    return A


def grad_NLL(g_i, N_i, Ntot, M_l, W_il):
    """Computes gradient of negative log-likelihood objective function.

    Args:
        g_i (np.array of shape (S-1,)): Array of total free energies associated with the windows
            0, 1, 2, ..., S-2. Free energy of the last window is fixed at 0 (reference value).
        N_i (np.array of shape (S-1,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-2.
        M_l (np.array of shape (M,)): Array of total bin counts for each bin.
        W_il (np.array of shape (S-1, M)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-2.

    Returns:
        grad_g A (np.float): Gradient of negative log-likelihood objective function w.r.t. g.
    """
    g_i_mat = np.repeat(g_i[:, np.newaxis], len(M_l), axis=1)
    N_i_mat = np.repeat(N_i[:, np.newaxis], len(M_l), axis=1)

    return N_i * np.exp(g_i) * np.sum(np.exp(np.log(M_l) + W_il - logsumexp(np.exp(g_i_mat + W_il), b=N_i_mat, axis=0)))


def minimize_NLL_solver(N_i, M_l, W_il, opt_method='BFGS'):
    """Computes optimal parameters g_i by minimizing the negative log-likelihood
    for jointly observing the bin counts in the indepedent windows in the dataset.

    Args:
        N_i (np.array of shape (S,)): Array of total sample counts for the windows
            0, 1, 2, ..., S-1.
        M_l (np.array of shape (M,)): Array of total bin counts for each bin.
        W_il (np.array of shape (S, M)): Array of weights, W_il = -beta * U_i(x_l) for the windows
            0, 1, 2, ..., S-1.
        opt_method (string): Optimization algorithm to use (see options supported by scipy.optimize.minimize)

    Returns:
        (g_i (np.array of shape(S,)), status (bool, solution status))
    """
    S = len(N_i)
    M = len(M_l)
    Ntot = np.sum(N_i)

    g_opt_0 = np.zeros((S - 1))  # Initial guess

    # Optimize
    res = scipy.optimize.minimize(NLL, g_opt_0, args=(N_i[:-1], Ntot, M_l, W_il[:-1, :]), method=opt_method)

    g_opt = res.x
    g_i = np.zeros(S)
    g_i[:-1] = g_opt

    return g_i, res.status


def self_consistent_solver(N_i, M_l, W_il, tol=1e-7, maxiter=100000, logevery=None):
    """Computes optimal parameters g_i by solving the coupled WHAM equations self-consistently
    until convergence.

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
    S = len(N_i)
    M = len(M_l)

    g_i = 1e-16 * np.ones(S)

    # Solution obtained with set  tolerance level?
    status = False

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

        if logevery is not None:
            if iter % logevery == 0:
                print("Self-consistent solver error = {:.2e}.".format(tol_check))

        if increment[np.argmax(increment)] < tol:
            if logevery is not None:
                print("Self-consistent solver error = {:.2e}.".format(tol_check))
            status = True
            break

    return g_i, status


def compute_betaF_profile(x_it, x_l, u_i, beta, solver='log-likelihood', scale_stat_ineff=False, **solverkwargs):
    """Computes the binned free energy profile.

    Args:
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        x_l (np.array): Array of bin centers of length M.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        solver (string): Solution technique to use ['log-likelihood', 'self-consistent', default='log-likelihood'].
        scale_stat_ineff (boolean): Compute and scale bin count by statistical inefficiency (default=False).
        **solverkwargs: Arguments for solver

    Returns:
        betaF_l (np.array): Array of WHAM/consensus free energies of length M.
        betaF_il (list(np.array)): S-length list of biased free energy profiles (M-length np.arrays) for each window.
    """
    S = len(u_i)
    M = len(x_l)

    # Compute bin edges
    bin_edges = np.zeros(M + 1)

    # add an artificial edge before the first center
    bin_edges[0] = x_l[0] - (x_l[1] - x_l[0]) / 2

    # add an artificial edge after the last center
    bin_edges[-1] = x_l[-1] + (x_l[-1] - x_l[-2]) / 2

    # bin edges are at the midpoints of bin centers
    bin_edges[1:-1] = (x_l[1:] + x_l[:-1]) / 2

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

    """Debug
    print M_l
    """

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

    """Debug
    # Compare to reference
    g_i = np.array([0, 2.272978, 7.433105, 15.85212, 26.12359, 35.60421, 44.90448, 56.14338])

    g_i_mat = np.repeat(g_i[:, np.newaxis], M, axis=1)
    N_i_mat = np.repeat(N_i[:, np.newaxis], M, axis=1)

    # Compute consensus/WHAM free energy profile
    betaF_l_ref = np.log(delta_x_l) + logsumexp(g_i_mat + W_il, b=N_i_mat, axis=0) - np.log(M_l)

    return betaF_l, betaF_l_ref, betaF_il
    """
    return betaF_l


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
        betaF_l_sample = compute_betaF_profile(x_it_bsample, x_l, u_i, beta, solver=solver, scale_stat_ineff=False, **solverkwargs)

        # Append to samples
        betaF_l_all_samples[:, sidx] = betaF_l_sample

    # Calculate mean
    betaF_l = betaF_l_all_samples.mean(axis=1)

    # Calculate bootstrap error
    betaF_l_berr = betaF_l_all_samples.std(axis=1) / np.sqrt(Nboot)

    return betaF_l, betaF_l_berr
