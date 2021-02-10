"""Defines functions for dealing with correlated timeseries data."""
import numpy as np
from statsmodels.tsa import stattools


def statisticalInefficiency(x, fft=True):
    """Computes the statistical inefficiency of x.

    Args:
        x (ndarray): 1-dimensional array containing correlated data.
        fft (bool): Use fft when computing autocorrelation function (default=True).

    Returns:
        g (float): Statistical inefficiency of x."""
    acf = stattools.acf(x, nlags=len(x), fft=fft)
    acf_cross_point = np.argmax(acf < 0) - 1

    t = np.array(range(len(x[:acf_cross_point])))
    T = len(x)
    tau = np.sum(np.multiply(np.array(1 - t / T), acf[:acf_cross_point]))
    g = 1 + 2 * tau

    return g


def bootstrap_independent_sample(x, g=None):
    """Draws an independent sample of size N_ind = N/g from
    x. If the statistical inefficiency g is not passed as a parameter,
    it will be calculated first.

    Args:
        x (ndarray): 1-dimensional array of length N containing correlated data to draw (uncorrelated) sample from.
        g (float): Statistical inefficiency (default=None).

    Returns:
        y (ndarray): 1-dimensional array of length N/g containing random samples drawn from x (with replacement)."""
    if g is None:
        g = statisticalInefficiency(x)

    return np.random.choice(x, size=int(len(x) / g), replace=True)


def bootstrap_window_samples(x_it, g=None):
    """Draws a bootstrap sample from each window.

    Args:
        x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
            data from the i'th window.
        g (float): Statistical inefficiency (default=None).

    Returns:
        x_it_b (list): Nested list of length S, x_it_b[i] is a 1-dimensional array of
            length N/g containing random samples drawn from x_it[i] (with replacement)."""
    x_it_b = []
    for x in x_it:
        x_it_b.append(bootstrap_independent_sample(x, g))
    return x_it_b
