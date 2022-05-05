"""
Calculates free energy profiles from test 1D and 2D umbrella sampling data using binned WHAM.
"""
import inspect
from multiprocessing import Pool
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pytest

import WHAM.binless
import WHAM.statistics
from WHAM.lib import timeseries, potentials


matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def arrange():
    """Prepares temp directories for test output."""
    if not os.path.exists("test_out"):
        os.makedirs("test_out")


################################################################################
# 1D tests
################################################################################


def get_data():
    pass


def test_binned_self_consistent():
    """
    Compares 1D free energy profile computed by self-consistent WHAM against analytical landscape.
    Checks that KLD < 0.01 across umbrellas.
    """
    # 1. Test KL-divergences

    # 2. Test L2-relative error b/w profile and true landscape
    pass


def test_binned_log_likelihood():
    """
    Compares 1D free energy profile computed by log-likelihood WHAM against analytical landscape.
    Checks that KLD < 0.01 across umbrellas.
    """
    # 1. Test phi-ensemble v/s gradient

    # 2. Test phi* v/s gradient
    pass


def test_binned_self_consistent_phi_ensemble():
    """
    Compares variance <dx^2> in a phi-ensemble against the derivative of the mean d<x>/dphi,
    where <x> and <x^2> are computed from self-consistent WHAM free energy profiles. Compares peak in susceptibility to known peak.
    """
    # 1. Test KL-divergences

    # 2. Test L2-relative error b/w profile and true landscape
    pass


def test_binned_log_likelihood_phi_ensemble():
    """
    Compares variance <dx^2> in a phi-ensemble against the derivative of the mean d<x>/dphi,
    where <x> and <x^2> are computed from log-likelihood WHAM free energy profiles. Compares peak in susceptibility to known peak.
    """
    # 1. Test phi-ensemble v/s gradient

    # 2. Test phi* v/s gradient
    pass


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
