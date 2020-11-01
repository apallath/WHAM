"""
Calculates free energy profile of test INDUS data using binned WHAM.
"""
import sys
import inspect
import re

import numpy as np
import matplotlib.pyplot as plt

from WHAM.lib.potentials import harmonic
import WHAM.binned


def get_test_data():
    # N* associated with each window
    n_star_win = [30, 25, 20, 15, 10, 5, 0, -5]
    # kappa associated with each window
    kappa_win = [0.98] * len(n_star_win)
    # Umbrella potentials (applied on CV) from each window
    # In this case, the potential was a harmonic potential, kappa/2 (N - N*)^2
    # For the unbiased window, the umbrella potential is 0.
    umbrella_win = [lambda x: 0]
    for i in range(1, len(n_star_win)):
        kappa = kappa_win[i]
        n_star = n_star_win[i]
        umbrella_win.append(harmonic(kappa, n_star))

    # List of bins to perform binning into
    bin_points = np.linspace(0, 34, 34 + 1)

    # Raw, correlated timeseries CV data from each window
    Ntw_win = []

    # Parse data from test files
    with open('test_data/unbiased/time_samples.out') as f:
        Ntw_i = []
        for line in f:
            if line.strip()[0] != '#':
                vals = line.strip().split()
                Ntw_i.append(float(vals[2]))
        Ntw_i = np.array(Ntw_i)
        Ntw_win.append(Ntw_i[500:])

    for nstar in ['25.0', '20.0', '15.0', '10.0', '5.0', '0.0', '-5.0']:
        with open('test_data/nstar_{}/plumed.out'.format(nstar)) as f:
            Ntw_i = []
            for line in f:
                if line.strip()[0] != '#':
                    vals = line.strip().split()
                    Ntw_i.append(float(vals[2]))
            Ntw_i = np.array(Ntw_i)
            Ntw_win.append(Ntw_i[200:])

    beta = 1000 / (8.314 * 300)  # at 300 K, in kJ/mol units

    return n_star_win, Ntw_win, bin_points, umbrella_win, beta


def test_binned_Nt_self_consistent():
    n_star_win, Ntw_win, bin_points, umbrella_win, beta = get_test_data()

    # Perform WHAM calculation
    betaF_l, betaF_il, g_i, status = WHAM.binned.compute_betaF_profile(Ntw_win, bin_points,
                                                                       umbrella_win, beta, bin_style='left',
                                                                       solver='self-consistent', scale_stat_ineff=True,
                                                                       tol=1e-12, logevery=100)  # solver kwargs

    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_l = betaF_l - betaF_l[30]  # reposition zero so that unbiased free energy is zero

    # Plot consensus
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_points, betaF_l, label="Self-consistent WHAM solution")

    bin_points_ref = np.load("seanmarks_ref/bins.npy")
    betaF_l_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_points_ref, betaF_l_ref, label="seanmarks/wham (binless)")
    ax.legend()
    plt.savefig("self_consistent.png")

    # Plot window
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    for i in range(betaF_il.shape[0]):
        ax.plot(bin_points, betaF_il[i], label=r"$N^*$ = {}".format(n_star_win[i]))
    ax.legend()
    plt.savefig("self_consistent_biased.png")


def test_binned_Nt_log_likelihood():
    n_star_win, Ntw_win, bin_points, umbrella_win, beta = get_test_data()

    # Perform WHAM calculation
    betaF_l, betaF_il, g_i, status = WHAM.binned.compute_betaF_profile(Ntw_win, bin_points,
                                                                       umbrella_win, beta, bin_style='left',
                                                                       solver='log-likelihood', scale_stat_ineff=True,
                                                                       opt_method='BFGS')  # solver kwargs

    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_l = betaF_l - betaF_l[30]  # reposition zero so that unbiased free energy is zero

    # Plot consensus
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_points, betaF_l, label="Log-likelihood WHAM solution")

    bin_points_ref = np.load("seanmarks_ref/bins.npy")
    betaF_l_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_points_ref, betaF_l_ref, label="seanmarks/wham (binless)")
    ax.legend()
    plt.savefig("log_likelihood.png")

    # Plot window
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    for i in range(betaF_il.shape[0]):
        ax.plot(bin_points, betaF_il[i], label=r"$N^*$ = {}".format(n_star_win[i]))
    ax.legend()
    plt.savefig("log_likelihood_biased.png")


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
