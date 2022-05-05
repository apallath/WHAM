"""Unit tests for WHAM.lib.timeseries"""
import sys
import inspect
import re

import numpy as np

from WHAM.lib import potentials


def test_harmonic_potential():
    x = np.array([1, 2, 3, 4, 5])
    u = potentials.harmonic(2, 2)
    print(u(x))
    assert(np.allclose(u(x), np.array([1, 0, 1, 4, 9])))


def test_linear_potential():
    x = np.array([1, 2, 3, 4, 5])
    u = potentials.linear(2)
    print(u(x))
    assert(np.allclose(u(x), np.array([2, 4, 6, 8, 10])))


def test_harmonic_DD_potential():
    # 1-d test
    x = np.array([1, 2, 3, 4, 5])
    u = potentials.harmonic_DD([2], [2])
    print(u(x))
    assert(np.allclose(u(x), np.array([1, 0, 1, 4, 9])))

    # 2-d test
    x = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6]])
    kappa = np.array([2, -2])
    xstar = np.array([2, 1])
    u = potentials.harmonic_DD(kappa, xstar)
    print(u(x))

    u_x = np.array([1 - 1, 0 - 4, 1 - 9, 4 - 16, 9 - 25])
    assert(np.allclose(u(x), u_x))


def test_linear_DD_potential():
    pass


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
