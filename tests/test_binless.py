"""
Calculates free energy profile of test INDUS data using binless WHAM
"""
import sys
import inspect
import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

from WHAM.lib import potentials
import WHAM.binless
import WHAM.statistics

import matplotlib
matplotlib.use('Agg')


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
    calc = WHAM.binless.Calc1D()
    betaF_bin, status = calc.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                   bin_style='left', solver='self-consistent',
                                                   tol=1e-2, logevery=100)
    g_i = calc.g_i

    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_bin = betaF_bin - betaF_bin[30]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_points, betaF_bin, 'x-', label="Self-consistent binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_bin_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_bin_ref, 'x-', label="seanmarks/wham (binless)")

    ax.legend()

    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F$")
    plt.savefig("test_out/binless_self_consistent_incomplete.png")

    """Reweighting check"""
    betaF_il, _ = WHAM.statistics.win_betaF(Ntw_win, bin_points, umbrella_win, beta,
                                            bin_style='left')
    betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, umbrella_win,
                                                                     beta, bin_style='left')
    # Plot window free energies
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    for i in range(betaF_il.shape[0]):
        ax.plot(bin_points, betaF_il[i], 'x--', label=r"$N^*$ = {}".format(n_star_win[i]), color="C{}".format(i))
        ax.plot(bin_points, betaF_il_reweight[i], color="C{}".format(i))
    ax.legend()
    ax.set_xlim([0, 35])
    ax.set_ylim([0, 8])

    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F_{bias, i}$")
    plt.savefig("test_out/binless_reweight_win_self_consistent_incomplete.png")

    # KL divergence check
    D_KL_i = WHAM.statistics.binless_KLD_reweighted_win_betaF(calc, Ntw_win, bin_points,
                                                              umbrella_win, beta, bin_style='left')
    print(D_KL_i)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(n_star_win, D_KL_i, 's')
    ax.axhline(y=0.1)
    ax.axhline(y=-0.1)
    ax.set_xlabel(r"$N^*$")
    ax.set_ylabel(r"$D_{KL}$")
    plt.savefig("test_out/binless_reweight_KLD_self_consistent_incomplete.png")

    # Save free energy
    np.save("test_out/binless_scf_gi.npy", g_i)

    #######################
    # Restart             #
    #######################

    print("Restarting from saved g_i's")

    g_i = np.load("test_out/binless_scf_gi.npy")

    # Perform WHAM calculation
    betaF_bin, status = calc.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                   bin_style='left', solver='self-consistent',
                                                   g_i=g_i, tol=1e-12, logevery=100)
    g_i = calc.g_i

    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_bin = betaF_bin - betaF_bin[30]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_points, betaF_bin, 'x-', label="Self-consistent binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_bin_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_bin_ref, 'x-', label="seanmarks/wham (binless)")

    ax.legend()

    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F$")
    plt.savefig("test_out/binless_self_consistent.png")

    """Reweighting check"""
    betaF_il, _ = WHAM.statistics.win_betaF(Ntw_win, bin_points, umbrella_win, beta,
                                            bin_style='left')
    betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, umbrella_win,
                                                                     beta, bin_style='left')
    # Plot window free energies
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    for i in range(betaF_il.shape[0]):
        ax.plot(bin_points, betaF_il[i], 'x--', label=r"$N^*$ = {}".format(n_star_win[i]), color="C{}".format(i))
        ax.plot(bin_points, betaF_il_reweight[i], color="C{}".format(i))
    ax.legend()
    ax.set_xlim([0, 35])
    ax.set_ylim([0, 8])

    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F_{bias, i}$")
    plt.savefig("test_out/binless_reweight_win_self_consistent.png")

    # KL divergence check
    D_KL_i = WHAM.statistics.binless_KLD_reweighted_win_betaF(calc, Ntw_win, bin_points,
                                                              umbrella_win, beta, bin_style='left')
    print(D_KL_i)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(n_star_win, D_KL_i, 's')
    ax.axhline(y=0.1)
    ax.axhline(y=-0.1)
    ax.set_xlabel(r"$N^*$")
    ax.set_ylabel(r"$D_{KL}$")
    plt.savefig("test_out/binless_reweight_KLD_self_consistent.png")

    # KL_D should be low
    assert(np.all(np.abs(D_KL_i) < 0.1))

    # Benchmark against seanmarks/wham
    assert(np.max(np.abs(betaF_bin - betaF_bin_ref)) < 1)
    assert(np.sum(np.sqrt((betaF_bin - betaF_bin_ref) ** 2)) < 2)


def test_binless_log_likelihood():
    n_star_win, Ntw_win, bin_points, umbrella_win, beta = get_test_data()

    # Perform WHAM calculation
    calc = WHAM.binless.Calc1D()
    betaF_bin, status = calc.compute_betaF_profile(Ntw_win, bin_points, umbrella_win, beta,
                                                   bin_style='left', solver='log-likelihood',
                                                   logevery=1)
    g_i = calc.g_i

    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_bin = betaF_bin - betaF_bin[30]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(bin_points, betaF_bin, 'x-', label="Log-likelihood binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_bin_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_bin_ref, 'x-', label="seanmarks/wham (binless)")

    ax.legend()

    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F$")
    plt.savefig("test_out/binless_log_likelihood.png")

    """Reweighting check"""
    betaF_il, _ = WHAM.statistics.win_betaF(Ntw_win, bin_points, umbrella_win, beta,
                                            bin_style='left')
    betaF_il_reweight = WHAM.statistics.binless_reweighted_win_betaF(calc, bin_points, umbrella_win,
                                                                     beta, bin_style='left')
    # Plot window free energies
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    for i in range(betaF_il.shape[0]):
        ax.plot(bin_points, betaF_il[i], 'x--', label=r"$N^*$ = {}".format(n_star_win[i]), color="C{}".format(i))
        ax.plot(bin_points, betaF_il_reweight[i], color="C{}".format(i))
    ax.legend()
    ax.set_xlim([0, 35])
    ax.set_ylim([0, 8])

    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"$\beta F_{bias, i}$")
    plt.savefig("test_out/binless_reweight_win_log_likelihood.png")

    # KL divergence check
    D_KL_i = WHAM.statistics.binless_KLD_reweighted_win_betaF(calc, Ntw_win, bin_points,
                                                              umbrella_win, beta, bin_style='left')
    print(D_KL_i)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(n_star_win, D_KL_i, 's')
    ax.axhline(y=0.1)
    ax.axhline(y=-0.1)
    ax.set_xlabel(r"$N^*$")
    ax.set_ylabel(r"$D_{KL}$")
    plt.savefig("test_out/binless_reweight_KLD_log_likelihood.png")

    # KL_D should be low
    assert(np.all(np.abs(D_KL_i) < 0.1))

    # Benchmark against seanmarks/wham
    assert(np.max(np.abs(betaF_bin - betaF_bin_ref)) < 1)
    assert(np.sum(np.sqrt((betaF_bin - betaF_bin_ref) ** 2)) < 2)


def get_2D_test_data():
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
    x_bin_points = np.linspace(0, 34, 34 + 1)
    y_bin_points = np.linspace(0, 34, 34 + 1)

    # Raw, correlated timeseries CV data from each window
    Ntw_win = []
    N_win = []

    # Parse data from test files
    with open('test_data/unbiased/time_samples.out') as f:
        Ntw_i = []
        N_i = []
        for line in f:
            if line.strip()[0] != '#':
                vals = line.strip().split()
                Ntw_i.append(float(vals[2]))
                N_i.append(float(vals[1]))
        Ntw_i = np.array(Ntw_i)
        Ntw_win.append(Ntw_i[500:])

        N_i = np.array(N_i)
        N_win.append(N_i[500:])

    for nstar in ['25.0', '20.0', '15.0', '10.0', '5.0', '0.0', '-5.0']:
        with open('test_data/nstar_{}/plumed.out'.format(nstar)) as f:
            Ntw_i = []
            N_i = []
            for line in f:
                if line.strip()[0] != '#':
                    vals = line.strip().split()
                    Ntw_i.append(float(vals[2]))
                    N_i.append(float(vals[1]))

            Ntw_i = np.array(Ntw_i)
            Ntw_win.append(Ntw_i[500:])

            N_i = np.array(N_i)
            N_win.append(N_i[500:])

    beta = 1000 / (8.314 * 300)  # at 300 K, in kJ/mol units

    return n_star_win, Ntw_win, N_win, x_bin_points, y_bin_points, umbrella_win, beta


def test_binless_2D_log_likelihood():
    n_star_win, Ntw_win, N_win, x_bin_points, y_bin_points, umbrella_win, beta = get_2D_test_data()

    # Unroll Ntw_win into a single array
    x_l = Ntw_win[0]
    for i in range(1, len(Ntw_win)):
        x_l = np.hstack((x_l, Ntw_win[i]))

    # Unroll N_win into a single array
    y_l = N_win[0]
    for i in range(1, len(N_win)):
        y_l = np.hstack((y_l, N_win[i]))

    N_i = np.array([len(arr) for arr in Ntw_win])

    # Perform WHAM calculation
    calc = WHAM.binless.Calc1D()
    status = calc.compute_point_weights(x_l, N_i, umbrella_win, beta,
                                        solver='log-likelihood',
                                        logevery=1)
    g_i = calc.g_i

    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_2D_bin, _ = calc.bin_2D_betaF_profile(y_l, x_bin_points, y_bin_points)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    im_bin = ax[0].imshow(betaF_2D_bin, extent=[x_bin_points[0], x_bin_points[-1], y_bin_points[-1], y_bin_points[0]], cmap='cool')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_bin, cax=cax, orientation='vertical')

    ax[0].set_xlabel(r"$\tilde{N}$")
    ax[0].set_ylabel(r"N")
    ax[0].set_title("WHAM")

    betaF_2D_ref = np.load("seanmarks_ref/2D_profile.npy")
    im_ref = ax[1].imshow(betaF_2D_ref, cmap='cool')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_ref, cax=cax, orientation='vertical')

    ax[1].set_xlabel(r"$\tilde{N}$")
    ax[1].set_ylabel(r"N")
    ax[1].set_title("seanmarks/wham")

    plt.savefig("test_out/binless_2D_log_likelihood.png")

    # Difference plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    diff = betaF_2D_bin - betaF_2D_ref

    offset = mcolors.TwoSlopeNorm(vmin=-0.5,
                                  vcenter=0., vmax=2)
    im_bin = ax.imshow(diff, extent=[x_bin_points[0], x_bin_points[-1], y_bin_points[-1], y_bin_points[0]], cmap='bwr', norm=offset)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_bin, cax=cax, orientation='vertical')

    ax.set_xlabel(r"$\tilde{N}$")
    ax.set_ylabel(r"N")
    ax.set_title("WHAM - seanmarks/wham")

    plt.savefig("test_out/binless_2D_log_likelihood_diff.png")

    """Integrate out N~ to get free energy profile in N"""
    betaF_N = calc.bin_second_betaF_profile(y_l, x_bin_points, y_bin_points)

    betaF_N = betaF_N - betaF_N[30]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(y_bin_points, betaF_N, 'x-', label="Self-consistent binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_N_ref = np.load("seanmarks_ref/betaF_N.npy")
    ax.plot(bin_centers_ref, betaF_N_ref, 'x-', label="seanmarks/wham (binless)")

    ax.legend()

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\beta F$")
    plt.savefig("test_out/binless_2D_1D_N.png")


"""Alternate binning patterns"""

def get_2D_test_data_halfbin():
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
    x_bin_points = np.linspace(0, 34, 36 + 1)
    y_bin_points = np.linspace(0, 34, 17 + 1)

    # Raw, correlated timeseries CV data from each window
    Ntw_win = []
    N_win = []

    # Parse data from test files
    with open('test_data/unbiased/time_samples.out') as f:
        Ntw_i = []
        N_i = []
        for line in f:
            if line.strip()[0] != '#':
                vals = line.strip().split()
                Ntw_i.append(float(vals[2]))
                N_i.append(float(vals[1]))
        Ntw_i = np.array(Ntw_i)
        Ntw_win.append(Ntw_i[500:])

        N_i = np.array(N_i)
        N_win.append(N_i[500:])

    for nstar in ['25.0', '20.0', '15.0', '10.0', '5.0', '0.0', '-5.0']:
        with open('test_data/nstar_{}/plumed.out'.format(nstar)) as f:
            Ntw_i = []
            N_i = []
            for line in f:
                if line.strip()[0] != '#':
                    vals = line.strip().split()
                    Ntw_i.append(float(vals[2]))
                    N_i.append(float(vals[1]))

            Ntw_i = np.array(Ntw_i)
            Ntw_win.append(Ntw_i[500:])

            N_i = np.array(N_i)
            N_win.append(N_i[500:])

    beta = 1000 / (8.314 * 300)  # at 300 K, in kJ/mol units

    return n_star_win, Ntw_win, N_win, x_bin_points, y_bin_points, umbrella_win, beta


def test_binless_2D_log_likelihood_halfbin():
    n_star_win, Ntw_win, N_win, x_bin_points, y_bin_points, umbrella_win, beta = get_2D_test_data_halfbin()

    # Unroll Ntw_win into a single array
    x_l = Ntw_win[0]
    for i in range(1, len(Ntw_win)):
        x_l = np.hstack((x_l, Ntw_win[i]))

    # Unroll N_win into a single array
    y_l = N_win[0]
    for i in range(1, len(N_win)):
        y_l = np.hstack((y_l, N_win[i]))

    N_i = np.array([len(arr) for arr in Ntw_win])

    # Perform WHAM calculation
    calc = WHAM.binless.Calc1D()
    status = calc.compute_point_weights(x_l, N_i, umbrella_win, beta,
                                        solver='log-likelihood',
                                        logevery=1)
    g_i = calc.g_i

    # Optimized?
    print(status)

    # Useful for debugging:
    print("Window free energies: ", g_i)

    betaF_2D_bin, _ = calc.bin_2D_betaF_profile(y_l, x_bin_points, y_bin_points)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    im_bin = ax[0].imshow(betaF_2D_bin, extent=[x_bin_points[0], x_bin_points[-1], y_bin_points[-1], y_bin_points[0]], cmap='cool')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_bin, cax=cax, orientation='vertical')

    ax[0].set_xlabel(r"$\tilde{N}$")
    ax[0].set_ylabel(r"N")
    ax[0].set_title("WHAM")

    betaF_2D_ref = np.load("seanmarks_ref/2D_profile.npy")
    im_ref = ax[1].imshow(betaF_2D_ref, cmap='cool')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_ref, cax=cax, orientation='vertical')

    ax[1].set_xlabel(r"$\tilde{N}$")
    ax[1].set_ylabel(r"N")
    ax[1].set_title("seanmarks/wham")

    plt.savefig("test_out/binless_2D_log_likelihood_halfbin.png")

    """Integrate out N~ to get free energy profile in N"""
    # Preprocess inf and nan
    betaF_N = calc.bin_second_betaF_profile(y_l, x_bin_points, y_bin_points)

    betaF_N = betaF_N - betaF_N[15]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(y_bin_points, betaF_N, 'x-', label="Self-consistent binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_N_ref = np.load("seanmarks_ref/betaF_N.npy")
    ax.plot(bin_centers_ref, betaF_N_ref, 'x-', label="seanmarks/wham (binless)")

    ax.legend()

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\beta F$")
    plt.savefig("test_out/binless_2D_1D_N_halfbin.png")


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
