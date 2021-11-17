"""Functions for checking consistency of WHAM calculations."""
import numpy as np
from scipy.special import logsumexp

from WHAM.lib.timeseries import statisticalInefficiency
from WHAM.lib import potentials


def D_KL(betaF_P, betaF_Q, delta_x_bin):
    """Computes the KL divergence between two probability distributions P
    and Q corresponding to free energy profiles betaF_P and betaF_Q.

    Args:
        betaF_P (ndarray): Free energy profile of length M.
        betaF_Q (ndarray): Free energy profile of length M.
        delta_x_bin (ndarray): Bin interval, length M.

    Returns:
        KL divergence (float)"""

    np.errstate(divide='ignore')

    return (delta_x_bin * np.exp(-betaF_P) * (betaF_Q - betaF_P)).sum()

###############################
# Binless WHAM checks         #
###############################


def win_betaF(x_it, x_bin, u_i, beta, bin_style='left', scale_stat_ineff=False):
    """Computes free energy profiles for each window.

    Args:
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        x_bin (list): Array of bin left-edges/centers of length M. Used only for computing final PMF.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.

    Returns:
        tuple(betaF_il, delta_x_bin)
            - betaF_il (np.array): 2-D array of biased free energies for each window, of shape (S, M)
            - delta_x_bin (np.array): Bin interval
    """
    np.errstate(divide='ignore')

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

    # Compute statistical inefficiencies if required, else set as 1 (=> no scaling)
    # For now, use pymbar instead of WHAM.lib.timeseries function
    stat_ineff = np.ones(S)
    if scale_stat_ineff:
        for i in range(S):
            stat_ineff[i] = statisticalInefficiency(x_it[i], fft=True)

    # Bin x_it
    n_il = np.zeros((S, M))
    for i in range(S):
        n_il[i, :], _ = np.histogram(x_it[i], bins=bin_edges)
        n_il[i, :] = n_il[i, :] / stat_ineff[i]

    N_i = n_il.sum(axis=1)

    # Compute biased free energy profiles for each window
    betaF_il = np.zeros((S, M))
    for i in range(S):
        p_il = n_il[i, :] / N_i[i]
        betaF_il[i, :] = -np.log(p_il / delta_x_bin)

    return betaF_il, delta_x_bin


def binned_reweighted_win_betaF(x_bin, betaF_bin, u_i, beta):
    """
    Computes free energy profile for each window i by reweighting the entire free
    energy profile betaF_bin at x_bin to the biased ensemble with bias potential
    specified by u_i[i]. This function can be used to reweight free energy profiles
    constructed using either binned or binless WHAM.

    Args:
        x_bin (list): Array of bin left-edges/centers of length M.
        betaF_bin (list): Array of free energies of length M.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).

    Returns:
        betaF_il_reweight (np.array): 2-D array of biased free energies for each window, of shape (S, M)
    """
    np.errstate(divide='ignore')

    S = len(u_i)
    M = len(x_bin)

    betaF_il_reweight = np.zeros((S, M))

    for i in range(S):
        ui_bin = u_i[i](np.array(x_bin))
        # Add bias
        betaF_il_reweight[i, :] = betaF_bin + beta * ui_bin
        # No need to normalize: handle by subtracting out minimum
        betaF_il_reweight[i, :] = betaF_il_reweight[i, :] - np.min(betaF_il_reweight[i, :])

    return betaF_il_reweight


def binless_reweighted_win_betaF(calc, x_bin, u_i, beta, bin_style='left'):
    """Computes free energy profiles for each window i by reweighting the entire
    free energy profile to the biased ensemble with bias potential specified
    by u_i[i].

    Args:
        calc (WHAM.binless.Calc1D): Binless Calc1D object, with weights pre-computed.
        x_bin (list): Array of bin left-edges/centers of length M. Used only for computing final PMF.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.

    Returns:
        betaF_il_reweight (np.array): 2-D array of biased free energies for each window, of shape (S, M)"""

    np.errstate(divide='ignore')

    S = len(u_i)
    M = len(x_bin)

    betaF_il_reweight = np.zeros((S, M))
    for i in range(S):
        G_l_reweight = calc.reweight(beta, u_i[i])
        betaF_il_reweight[i, :], _ = calc.bin_betaF_profile(x_bin, G_l=G_l_reweight, bin_style=bin_style)

    return betaF_il_reweight


def binless_KLD_reweighted_win_betaF(calc, x_it, x_bin, u_i, beta, bin_style='left'):
    """Computes the KL divergence between probability distributions corresponding to
    (sampled) bias free energy profile and reweighted biased free energy profile (constructed from consensus WHAM)
    for each window.

    Args:
        calc (WHAM.binless.Calc1D): Binless Calc1D object, with weights pre-computed.
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        x_bin (list): Array of bin left-edges/centers of length M. Used only for computing final PMF.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.

    Returns:
        D_KL_i (ndarray): Array of length S containing KL divergence for each window"""

    np.errstate(divide='ignore')

    betaF_il, delta_x_bin = win_betaF(x_it, x_bin, u_i, beta, bin_style=bin_style)
    betaF_il_reweight = binless_reweighted_win_betaF(calc, x_bin, u_i, beta, bin_style=bin_style)
    S = len(u_i)
    D_KL_i = np.zeros(S)
    for i in range(S):
        indices = np.where(betaF_il[i, :] < np.inf)
        D_KL_i[i] = D_KL(betaF_il[i, indices], betaF_il_reweight[i, indices], delta_x_bin[indices])

    return D_KL_i


###############################
# Binless WHAM phi-ensemble   #
###############################
def binless_reweight_phi_ensemble(calc, phi_vals, beta):
    """Computes <x_l> and <dx_l^2> v/s phi by reweighting the underlying free energy profile
    in calc to the phi-ensemble specified by phi_vals and beta.

    Args:
        calc (WHAM.binless.Calc1D): Binless Calc1D object, with weights pre-computed.
        phi_vals (ndarray): Array of phi values to compute <x_l> at, units of phi in kJ/mol.
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.

    Returns:
        N_avg_vals (ndarray): Array containing <x_l> corresponding to each value of phi.
        N_var_vals (ndarray): Array containing <dx_l^2> corresponding to each value of phi."""

    np.errstate(divide='ignore')

    N_avg_vals = np.zeros(len(phi_vals))
    N_var_vals = np.zeros(len(phi_vals))

    for i, phi in enumerate(phi_vals):
        u_phi = potentials.linear(phi)
        G_l_reweight = calc.reweight(beta, u_phi)
        # g_i cancels out during ensemble averaging => doesn't matter
        p_l_reweight = np.exp(-G_l_reweight)
        N_l = calc.x_l
        N_avg = np.average(N_l, weights=p_l_reweight)
        N_var = np.average((N_l - N_avg) ** 2, weights=p_l_reweight)
        N_avg_vals[i] = N_avg
        N_var_vals[i] = N_var

    return N_avg_vals, N_var_vals
