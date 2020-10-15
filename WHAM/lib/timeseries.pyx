"""Functions for dealing with correlated timeseries data"""
import numpy as np
from statsmodels.tsa import stattools


def statistical_inefficiency(x):
    """Computes the statistical inefficiency of x."""
    acf = stattools.acf(x, nlags=len(x), fft=True)
    acf_cross_point = np.argmax(acf < 0) - 1

    t = np.array(range(len(x[:acf_cross_point])))
    T = len(x)
    tau = np.sum(np.multiply(np.array(1 - t / T), acf[:acf_cross_point]))
    g = 1 + 2 * tau

    return g


def bootstrap_independent_sample(x, g=None):
    """Draws an independent sample of size N_ind = N/g from
    x. If the statistical inefficiency g is not passed as a parameter,
    it will be calculated first."""
    if g is None:
        g = statistical_inefficiency(x)

    return np.random.choice(x, size=int(len(x) / g))
