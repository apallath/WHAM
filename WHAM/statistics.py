"""Functions for checking consistency of WHAM calculations."""
import numpy as np

from WHAM.lib.timeseries import statisticalInefficiency
from WHAM.lib import potentials


################################################################################
# Helper functions
################################################################################

def D_KL(betaF_P, betaF_Q, delta_x_bin):
    """Computes the KL divergence between two probability distributions P
    and Q corresponding to free energy profiles betaF_P and betaF_Q.

    Args:
        betaF_P (ndarray): Free energy profile of length M.
        betaF_Q (ndarray): Free energy profile of length M.
        delta_x_bin (ndarray): Bin interval, length M.

    Returns:
        KL divergence (float)"""

    with np.errstate(divide='ignore'):
        return (delta_x_bin * np.exp(-betaF_P) * (betaF_Q - betaF_P)).sum()


def D_KL_DD(betaF_P, betaF_Q, delta_X_bin):
    raise NotImplementedError()


################################################################################
# Functions to reweight free energy profiles
################################################################################


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

    with np.errstate(divide='ignore'):
        S = len(u_i)
        M = len(x_bin)

        betaF_il_reweight = np.zeros((S, M))
        for i in range(S):
            G_l_reweight = calc.reweight(beta, u_i[i])
            betaF_il_reweight[i, :], _ = calc.bin_betaF_profile(x_bin, G_l=G_l_reweight, bin_style=bin_style)

        return betaF_il_reweight


def binless_reweighted_win_betaF_DD(calc, X_bin, u_i, beta, bin_style='left'):
    raise NotImplementedError()


def binned_reweighted_win_betaF(x_bin, betaF_bin, u_i, beta):
    """
    Computes free energy profile for each window i by reweighting the entire free
    energy profile betaF_bin at x_bin to the biased ensemble with bias potential
    specified by u_i[i].

    Note:
        This function can be also be used to reweight free energy profiles
        constructed using binless WHAM.

    Args:
        x_bin (list): Array of bin left-edges/centers of length M.
        betaF_bin (list): Array of free energies of length M.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).

    Returns:
        betaF_il_reweight (np.array): 2-D array of biased free energies for each window, of shape (S, M)
    """
    with np.errstate(divide='ignore'):
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


def binned_reweighted_win_betaF_DD(X_bin, betaF_bin, u_i, beta):
    raise NotImplementedError()


################################################################################
# Functions to compute phi-ensemble averages and susceptibilities in 1D
################################################################################

def binless_reweight_phi_ensemble(calc, phi_vals, beta):
    """Computes <x_l> and <dx_l^2> v/s phi by reweighting the underlying free energy profile
    in calc to the phi-ensemble specified by phi_vals and beta.

    Args:
        calc (WHAM.binless.Calc1D): Binless Calc1D object, with weights pre-computed.
        phi_vals (ndarray): Array of phi values to compute <x_l> at, units of phi (e.g. kJ/mol).
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.

    Returns:
        x_avg_vals (ndarray): Array containing <x_l> corresponding to each value of phi.
        x_var_vals (ndarray): Array containing <dx_l^2> corresponding to each value of phi."""

    with np.errstate(divide='ignore'):
        x_avg_vals = np.zeros(len(phi_vals))
        x_var_vals = np.zeros(len(phi_vals))

        for i, phi in enumerate(phi_vals):
            u_phi = potentials.linear(phi)
            G_l_reweight = calc.reweight(beta, u_phi)
            # g_i cancels out during ensemble averaging => doesn't matter
            p_l_reweight = np.exp(-G_l_reweight)
            x_l = calc.x_l
            x_avg = np.average(x_l, weights=p_l_reweight)
            x_var = np.average((x_l - x_avg) ** 2, weights=p_l_reweight)
            x_avg_vals[i] = x_avg
            x_var_vals[i] = x_var

        return x_avg_vals, x_var_vals


def binned_reweight_phi_ensemble(x_bin, betaF_bin, phi_vals, beta):
    """Computes <x_l> and <dx_l^2> v/s phi by reweighting the free energy profile
    betaF_bin to the phi-ensemble specified by phi_vals and beta.

    Args:
        x_bin (list): Array of bin left-edges/centers of length M.
        phi_vals (ndarray): Array of phi values to compute <x_l> at, units of phi (e.g. kJ/mol).
        betaF_bin (list): Array of free energies of length M.
        beta: beta, in inverse units to the units of u_i(x).

    Returns:
        x_avg_vals (ndarray): Array containing <x_l> corresponding to each value of phi.
        x_var_vals (ndarray): Array containing <dx_l^2> corresponding to each value of phi."""

    with np.errstate(divide='ignore'):
        x_avg_vals = np.zeros(len(phi_vals))
        x_var_vals = np.zeros(len(phi_vals))

        for i, phi in enumerate(phi_vals):
            u_phi = potentials.linear(phi)
            betaF_l_reweight = binned_reweighted_win_betaF(x_bin, betaF_bin, [u_phi], beta)[0]
            # g_i cancels out during ensemble averaging => doesn't matter
            p_l_reweight = np.exp(-betaF_l_reweight)
            x_l = x_bin
            x_avg = np.average(x_l, weights=p_l_reweight)
            x_var = np.average((x_l - x_avg) ** 2, weights=p_l_reweight)
            x_avg_vals[i] = x_avg
            x_var_vals[i] = x_var

        return x_avg_vals, x_var_vals


################################################################################
# Statistical checks: functions to compare raw biased free energy profiles
# against reweighted biased free energy profiles.
################################################################################

def win_betaF(x_it, x_bin, u_i, beta, bin_style='left', scale_stat_ineff=False):
    """Computes biased free energy profiles for each umbrella window from raw data.

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
    with np.errstate(divide='ignore'):
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


def win_betaF_DD(X_it, X_bin, u_i, beta, bin_style='left', scale_stat_ineff=False):
    raise NotImplementedError()


def binless_KLD_reweighted_win_betaF(calc, x_it, x_bin, u_i, beta, bin_style='left'):
    """Computes the KL divergence between probability distributions corresponding to
    (sampled) bias free energy profile and reweighted biased free energy profile (constructed by reweighting binless WHAM profile)
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

    with np.errstate(divide='ignore'):
        betaF_il, delta_x_bin = win_betaF(x_it, x_bin, u_i, beta, bin_style=bin_style)
        betaF_il_reweight = binless_reweighted_win_betaF(calc, x_bin, u_i, beta, bin_style=bin_style)
        S = len(u_i)
        D_KL_i = np.zeros(S)
        for i in range(S):
            indices = np.where(betaF_il[i, :] < np.inf)
            D_KL_i[i] = D_KL(betaF_il[i, indices], betaF_il_reweight[i, indices], delta_x_bin[indices])

        return D_KL_i


def binless_KLD_reweighted_win_betaF_DD(calc, X_it, X_bin, u_i, beta, bin_style='left'):
    raise NotImplementedError()


def binned_KLD_reweighted_win_betaF(x_it, x_bin, betaF_bin, u_i, beta, bin_style='left'):
    """Computes the KL divergence between probability distributions corresponding to
    (sampled) bias free energy profile and reweighted biased free energy profile (constructed by reweighting binned WHAM profile)
    for each window.

    Args:
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        x_bin (list): Array of bin left-edges/centers of length M.
        betaF_bin (list): Array of (consensus) free energies of length M.
        u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
            acting on the i'th window.
        beta: beta, in inverse units to the units of u_i(x).
        bin_style (string): 'left' or 'center'.

    Caution:
        For the best results, use the same bin style as the consensus free energy profile.

    Returns:
        D_KL_i (ndarray): Array of length S containing KL divergence for each window"""

    with np.errstate(divide='ignore'):
        betaF_il, delta_x_bin = win_betaF(x_it, x_bin, u_i, beta, bin_style=bin_style)
        betaF_il_reweight = binned_reweighted_win_betaF(x_bin, betaF_bin, u_i, beta)
        S = len(u_i)
        D_KL_i = np.zeros(S)
        for i in range(S):
            indices = np.where(betaF_il[i, :] < np.inf)
            D_KL_i[i] = D_KL(betaF_il[i, indices], betaF_il_reweight[i, indices], delta_x_bin[indices])

        return D_KL_i


def binned_KLD_reweighted_win_betaF_DD(X_it, X_bin, betaF_bin, u_i, beta, bin_style='left'):
    raise NotImplementedError()
