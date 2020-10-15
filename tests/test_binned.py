"""Calculates free energy profile from test INDUS data using binned WHAM"""
import sys
import inspect
import re

import numpy as np
import matplotlib.pyplot as plt

from WHAM.lib.potentials import harmonic
import WHAM.binned


def test_binned_Nt_self_consistent_stat_ineff():
    # N* associated with each window
    n_star_win = [30, 25, 20, 15, 10, 5, 0, -5]
    # kappa associated with each window
    kappa_win = [0.98] * len(n_star_win)
    # Umbrella potentials (applied on CV) from each window
    # In this case, the potential was a harmonic potential, kappa/2 (N - N*)^2
    umbrella_win = []
    for i in range(len(n_star_win)):
        kappa = kappa_win[i]
        n_star = n_star_win[i]
        umbrella_win.append(harmonic(kappa, n_star))

    # List of bins to perform binning into
    bin_centers = np.linspace(0, 34, 34 + 1)

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
            Ntw_win.append(Ntw_i[500:])

    for i in range(len(n_star_win)):
        print(np.min(Ntw_win[i]), np.max(Ntw_win[i]))

    beta = 1000 / (8.314 * 300)  # at 300 K, in kJ/mol units

    # Perform WHAM calculation
    betaF_l = WHAM.binned.compute_betaF_profile(Ntw_win, bin_centers,
                                                umbrella_win, beta, solver='self-consistent',
                                                scale_stat_ineff=True, tol=1e-12)

    betaF_l = betaF_l - betaF_l[30]  # resposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_centers, betaF_l, label="Solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_l_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_l_ref, label="Reference (Sean Marks)")
    ax.legend()
    plt.savefig("self_consistent_stat_ineff.png")


def test_binned_Nt_bootstrap():
    # N* associated with each window
    n_star_win = [30, 25, 20, 15, 10, 5, 0, -5]
    # kappa associated with each window
    kappa_win = [0.98] * len(n_star_win)
    # Umbrella potentials (applied on CV) from each window
    # In this case, the potential was a harmonic potential, kappa/2 (N - N*)^2
    umbrella_win = []
    for i in range(len(n_star_win)):
        kappa = kappa_win[i]
        n_star = n_star_win[i]
        umbrella_win.append(harmonic(kappa, n_star))

    # List of bins to perform binning into
    bin_centers = np.array(list(range(35)))

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
            Ntw_win.append(Ntw_i[500:])

    beta = 1000 / (8.314 * 300)  # at 300 K, in kJ/mol units

    # Perform WHAM calculation
    betaF_l, betaF_l_berr = WHAM.binned.bootstrap_betaF_profile(100, Ntw_win, bin_centers,
                                                                umbrella_win, beta, solver='self-consistent',
                                                                track_progress=True, tol=1e-12)

    betaF_l = betaF_l - betaF_l[30]  # resposition zero so that unbiased free energy is zero

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.errorbar(bin_centers, betaF_l, yerr=betaF_l_berr, capsize=3, label="Solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_l_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_l_ref, label="Reference (Sean Marks)")
    ax.legend()
    plt.savefig("self_consistent_bootstrap.png")


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
