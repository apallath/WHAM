import numpy as np
from WHAM.lib.timeseries import statisticalInefficiency

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
# Binless WHAM                #
###############################

def win_betaF(x_it, x_bin, u_i, beta, bin_style='left', scale_stat_ineff=False):
    """Compute free energy profiles for each window.

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
    M_l = n_il.sum(axis=0)

    # Raise exception if any of the bins have no data points
    if np.any(M_l == 0):
         raise Exception("Some bins are empty. Check for no-overlap regions or adjust binning so that no bins remain empty.")

    # Compute biased free energy profiles for each window
    betaF_il = np.zeros((S, M))
    for i in range(S):
        p_il = n_il[i, :] / N_i[i]
        betaF_il[i, :] = -np.log(p_il / delta_x_bin)

    return betaF_il, delta_x_bin


def binless_reweighted_win_betaF(calc, x_bin, u_i, beta, bin_style='left'):
    """Compute free energy profiles for each window i by reweighting the entire
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
        G_l_reweight = calc.reweight(beta, u_i[i], calc.g_i[i])
        betaF_il_reweight[i, :] = calc.bin_betaF_profile(x_bin, G_l=G_l_reweight, bin_style=bin_style)

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
