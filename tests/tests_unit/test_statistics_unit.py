"""Unit tests for WHAM.statistics"""
import inspect
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import WHAM.statistics

matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def arrange():
    """Prepares temp directories for test output."""
    if not os.path.exists("test_statistics_outputs"):
        os.makedirs("test_statistics_outputs")


def test_binned_reweight():
    Ntvals = []
    bFvals = []
    rNtvals = []
    rbFvals = []

    with open("test_statistics_inputs/betaF.dat", "r") as f:
        for line in f:
            if line.strip()[0] != '#':
                Nt, bF = tuple([float(i) for i in line.strip().split()])
                Ntvals.append(Nt)
                bFvals.append(bF)

    with open("test_statistics_inputs/betaF_phi_e_star.dat", "r") as f:
        for line in f:
            if line.strip()[0] != '#':
                rNt, rbF = tuple([float(i) for i in line.strip().split()])
                rNtvals.append(rNt)
                rbFvals.append(rbF)

    Ntvals = np.array(Ntvals)
    bFvals = np.array(bFvals)
    rNtvals = np.array(rNtvals)
    rbFvals = np.array(rbFvals)

    # Reweight Nt
    u = [lambda x: -0.09591 * x]
    beta = 1 / (8.314 * 298 / 1000)
    bFrvals = WHAM.statistics.binned_reweighted_win_betaF(Ntvals, bFvals, u, beta)[0]

    # Compare
    assert(np.max(rbFvals[100:226] - bFrvals[100:226]) < 1)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(rNtvals, rbFvals, label="Binless reweight")
    ax.plot(Ntvals, bFrvals, label="Binned reweight")
    ax.set_xlim([100, 225])
    ax.set_ylim([0, 6])
    plt.savefig("test_statistics_outputs/compare.png")


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
