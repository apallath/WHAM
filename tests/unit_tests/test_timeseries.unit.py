"""Unit tests for WHAM.lib.timeseries"""
import sys
import inspect
import re

import numpy as np

import WHAM.lib.timeseries


def test_statistical_inefficiency():
    # Stochastic test: may fail sometimes.
    pass


def test_bootstrap_indepedent_sample():
    # Stochastic test: may fail sometimes.
    pass


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
