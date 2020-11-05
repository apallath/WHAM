"""
Calculates free energy profile of test INDUS data using binless WHAM
"""
import sys
import inspect
import re

import numpy as np
import matplotlib.pyplot as plt

from WHAM.lib import potentials
import WHAM.binless


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
        umbrella_win.append(potentials.harmonic(kappa, n_star))

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


def test_binless_self_consistent():
    n_star_win, Ntw_win, bin_points, umbrella_win, beta = get_test_data()

    # Perform WHAM calculation
    betaF_bin, g_i, status = WHAM.binless.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                                bin_style='left', solver='self-consistent',
                                                                tol=1e-7, logevery=100)
    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_bin = betaF_bin - betaF_bin[30]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_points, betaF_bin, label="Self-consistent binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_bin_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_bin_ref, label="seanmarks/wham (binless)")
    ax.legend()
    plt.savefig("binless_self_consistent.png")

    # Save free energy
    np.save("binless_scf_gi.npy", g_i)


def test_binless_self_consistent_restart():
    n_star_win, Ntw_win, bin_points, umbrella_win, beta = get_test_data()

    print("Restarting from saved g_i's")

    g_i = np.load("binless_scf_gi.npy")

    # Perform WHAM calculation
    betaF_bin, g_i, status = WHAM.binless.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                                bin_style='left', solver='self-consistent',
                                                                g_i=g_i, tol=1e-12, logevery=100)
    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)


def test_binless_log_likelihood():
    n_star_win, Ntw_win, bin_points, umbrella_win, beta = get_test_data()

    # Perform WHAM calculation
    betaF_bin, g_i, status = WHAM.binless.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                                bin_style='left', solver='log-likelihood',
                                                                opt_method='BFGS')
    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_bin = betaF_bin - betaF_bin[30]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_points, betaF_bin, label="Log-likelihood binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_bin_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_bin_ref, label="seanmarks/wham (binless)")
    ax.legend()
    plt.savefig("binless_log_likelihood.png")


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
