"""
Calculates free energy profile of test INDUS data using binless WHAM

# IN PROGRESS!!!

"""
import sys
import inspect
import re

import numpy as np
import matplotlib.pyplot as plt

from WHAM.lib.potentials import harmonic
import WHAM.binless


def test_binned_Nt_self_consistent_stat_ineff():
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

    # Perform WHAM calculation
    x, betaF_l, status = WHAM.binless.compute_betaF_profile(Ntw_win, umbrella_win, beta, solver='self-consistent',
                                                            tol=1e-10)
    # Optimized?
    print(status)

    betaF_l = betaF_l - betaF_l[np.argmin((x - 30) ** 2)]  # reposition zero so that unbiased free energy is zero

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(x, betaF_l, label="Self-consistent binless WHAM solution")

    bin_centers_ref = np.load("seanmarks_ref/bins.npy")
    betaF_l_ref = np.load("seanmarks_ref/betaF_Ntilde.npy")
    ax.plot(bin_centers_ref, betaF_l_ref, label="Reference (Sean Marks)")
    ax.legend()
    plt.savefig("binless_self_consistent.png")


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
