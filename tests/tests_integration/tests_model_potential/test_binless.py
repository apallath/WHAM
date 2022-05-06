"""
Calculates free energy profiles from test 1D and 2D umbrella sampling data using binless WHAM.
"""
import inspect
from multiprocessing import Pool
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pytest

import WHAM.binless
import WHAM.statistics
from WHAM.lib import potentials


matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def arrange():
    """Prepares temp directories for test output."""
    if not os.path.exists("test_out"):
        os.makedirs("test_out")


################################################################################
# 1D tests
################################################################################


def read_traj(filename, skip=1):
    times = []
    traj = []

    count = 0
    with open(filename, 'r') as trajf:
        for line in trajf:
            if line.strip()[0] != '#':
                if count % skip == 0:
                    txyz = [float(c) for c in line.strip().split()]
                    times.append(txyz[0])
                    traj.append([txyz[1], txyz[2], txyz[3]])

    return np.array(times), np.array(traj)


def get_data():
    # constants
    kappa = 200
    x_min = -2
    x_max = 2
    delta_x = 0.2
    x_win = np.arange(x_min, x_max + delta_x, delta_x)

    # Data
    x_it = []
    y_it = []
    for x_0 in x_win:
        t, traj = read_traj("1D_US_test_data/biased/k{}x{:.2f}/traj.dat".format(kappa, x_0))
        x_it.append(traj[2000:, 0])
        y_it.append(traj[2000:, 1])

    # Umbrella potentials
    u_i = []
    for x_0 in x_win:
        u_i.append(potentials.harmonic(kappa, x_0))

    # Bins for free energy profile
    x_bin = np.linspace(-1.5, 1.5, 101)

    # Beta
    beta = 1000 / (8.3145 * 300)

    # Unroll - y
    y_l = y_it[0]
    for i in range(1, len(y_it)):
        y_l = np.hstack((y_l, y_it[i]))

    # Bins - y
    y_bin = np.linspace(-0.5, 0.5, 101)

    return x_it, x_bin, u_i, x_win, beta, y_l, y_bin


def get_data_2():
    # constants
    kappa = 200
    x_min = -2
    x_max = 2
    delta_x = 0.2

    x_win = np.arange(x_min, x_max + delta_x, delta_x)

    # Data
    x_it = []
    for x_0 in x_win:
        t, traj = read_traj("1D_US_test_data/biased/k{}x{:.2f}/traj.dat".format(kappa, x_0))
        x_it.append(traj[2000:, 0:2])

    # Umbrella potentials
    u_i = []
    for x_0 in x_win:
        u_i.append(potentials.harmonic_DD([kappa, 0], [x_0, 0]))

    # Bins for free energy profile
    x_bin_list = [np.linspace(-1.5, 1.5, 101), np.linspace(-0.5, 0.5, 101)]

    # Beta
    beta = 1000 / (8.3145 * 300)

    return x_it, x_bin_list, u_i, x_win, beta


def notest_binless_log_likelihood_1D():
    """
    1. Compares 1D free energy profile computed by log-likelihood WHAM against analytical landscape.
    Checks that KLD < 0.01 across umbrellas.

    2. Compares variance <dx^2> in a phi-ensemble against the derivative of the mean d<x>/dphi,
    where <x> and <x^2> are computed from log-likelihood WHAM free energy profiles. Compares peak in susceptibility to known peak.

    3. Reweights to 2D, integrates out x-coordinate, compares to analytical y-profile.
    """
    x_it, x_bin, u_i, x_win, beta, y_l, y_bin = get_data()

    calc = WHAM.binless.Calc1D()
    bF, _, _ = calc.compute_betaF_profile(x_it, x_bin, u_i, beta=beta)
    bF = bF - np.min(bF)

    # 1. Test KL-divergences
    D_KL_i = WHAM.statistics.binless_KLD_reweighted_win_betaF(calc, x_it, x_bin, u_i, beta)
    print(D_KL_i)

    fig, ax = plt.subplots(dpi=150)
    ax.plot(x_win, D_KL_i, 's')
    ax.axhline(y=0.01)
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$D_{KL}$")
    fig.savefig("test_out/1D_KLD.png")

    assert np.all(D_KL_i < 0.01)

    # 2. Test L2-relative error b/w profile and true landscape
    bF_true = beta * (30 * (x_bin ** 2 - 1) ** 2)
    L2_relative = np.sqrt(np.sum((bF - bF_true) ** 2)) / np.sqrt(np.sum(bF_true ** 2))
    print("L2 relative error b/w profile and analytical profile: {:.2f} pct".format(L2_relative * 100))

    fig, ax, = plt.subplots(dpi=150)
    ax.plot(x_bin, beta * (30 * (x_bin ** 2 - 1) ** 2), label="Analytical")
    ax.plot(x_bin, bF, label="WHAM")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 25)
    ax.set_ylabel(r"Free energy ($k_B T$)")
    ax.legend()
    fig.savefig("test_out/1D_profile_x.png")

    assert L2_relative < 0.06

    # 3. Reweight to phi-ensemble
    phivals = np.arange(-10, 10, 0.01)
    x_avg, x_var = WHAM.statistics.binless_reweight_phi_ensemble(calc, phivals, beta)
    susc = np.gradient(x_avg, phivals)

    fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=150)
    ax[0].plot(beta * phivals, x_avg)
    ax[0].set_xlabel(r"$\phi$")
    ax[0].set_ylabel(r"$\langle x \rangle$")
    ax[1].plot(beta * phivals, x_var, label=r"$\langle \delta x^2 \rangle$")
    ax[1].set_xlabel(r"$\phi$")
    ax[1].set_ylabel(r"$\langle \delta x^2 \rangle$")
    ax[1].plot(beta * phivals, -1 / beta * susc, label=r"$-\frac{1}{\beta} \frac{\partial \langle \tilde{N} \rangle}{\partial \beta \phi}$")
    ax[1].legend()
    fig.savefig("test_out/1D_phi_ensemble.png")

    phi_ens_L2_relative = np.sqrt(np.sum((-1 / beta * susc - x_var) ** 2)) / np.sqrt(np.sum(x_var ** 2))
    print("L2 relative error b/w gradient susc and var susc: {:.4f} pct".format(phi_ens_L2_relative * 100))

    assert phi_ens_L2_relative < 0.001

    # 4. Reweight to y
    bF_y = calc.bin_second_betaF_profile(y_l, x_bin, y_bin)
    bF_y = bF_y - np.min(bF_y)

    bF_y_true = beta * (30 * y_bin ** 2)
    L2_relative_y = np.sqrt(np.sum((bF_y - bF_y_true) ** 2)) / np.sqrt(np.sum(bF_y_true ** 2))
    print("L2 relative error b/w y-profile and analytical y-profile: {:.2f} pct".format(L2_relative_y * 100))

    fig, ax, = plt.subplots(dpi=150)
    ax.plot(x_bin, bF_y_true, label="Analytical")
    ax.plot(x_bin, bF_y, label="WHAM")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)
    ax.set_ylabel(r"Free energy ($k_B T$)")
    ax.legend()
    fig.savefig("test_out/1D_profile_rew_y.png")

    assert L2_relative_y < 0.1


def test_binless_log_likelihood_1D_CalcDD():
    x_it, x_bin_list, u_i, x_win, beta = get_data_2()

    calc = WHAM.binless.CalcDD()
    bF, bF_bin_counts, _ = calc.compute_betaF_profile(x_it, x_bin_list, u_i, beta=beta, bin_style='center')
    print(bF.shape)

    bF = bF - np.min(bF)

    np.set_printoptions(threshold=30)
    print(bF)
    print(bF_bin_counts)

    fig, ax = plt.subplots(dpi=150)
    levels = np.linspace(0, 25, 25)
    cmap = cm.jet
    contour_filled = ax.contourf(x_bin_list[0], x_bin_list[1], bF.T, levels, cmap=cm.get_cmap(cmap, len(levels) - 1))
    ax.contour(contour_filled, colors='k', alpha=0.5, linewidths=0.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(contour_filled, cax=cax, orientation='vertical')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.5, 0.5])
    cax.set_title(r"Free energy $(k_B T)$")
    fig.savefig("test_out/1D_DD_profile_xy.png")


################################################################################
# 2D tests
################################################################################


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
