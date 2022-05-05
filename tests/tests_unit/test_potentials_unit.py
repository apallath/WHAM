"""Unit tests for WHAM.lib.timeseries"""
import inspect
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from WHAM.lib import potentials


@pytest.fixture(autouse=True)
def arrange():
    """Prepares temp directories for test output."""
    if not os.path.exists("test_potentials_outputs"):
        os.makedirs("test_potentials_outputs")


def test_harmonic_potential():
    # analytical test
    x = np.array([1, 2, 3, 4, 5])
    u = potentials.harmonic(2, 2)
    print(u(x))
    assert(np.allclose(u(x), np.array([1, 0, 1, 4, 9])))


def test_linear_potential():
    # analytical test
    x = np.array([1, 2, 3, 4, 5])
    u = potentials.linear(2)
    print(u(x))
    assert(np.allclose(u(x), np.array([2, 4, 6, 8, 10])))


def test_harmonic_DD_potential():
    # 1-d analytical test
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    u = potentials.harmonic_DD([2], [2])
    print(u(x))
    assert(np.allclose(u(x), np.array([1, 0, 1, 4, 9])))

    # 2-d analytical test
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

    # visualization
    X, Y = np.mgrid[0:10:100j, 0:10:100j]
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    print(xy.shape)

    kappa = np.array([10, 2])
    xstar = np.array([5, 5])
    u = potentials.harmonic_DD(kappa, xstar)
    z = u(xy).reshape(100, 100)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    ax[0].contourf(X, Y, z)
    ax[1].contourf(X, Y, 5 * (X - 5) ** 2 + (Y - 5) ** 2)
    fig.savefig("test_potentials_outputs/harmonic_2D_test.png")

    assert(np.allclose(z, 5 * (X - 5) ** 2 + (Y - 5) ** 2))


def test_linear_DD_potential():
    # 1-d analytical test
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    u = potentials.linear_DD([2])
    print(u(x))
    assert(np.allclose(u(x), np.array([2, 4, 6, 8, 10])))

    # 2-d analytical test
    x = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6]])
    phi = np.array([2, -2])
    u = potentials.linear_DD(phi)
    print(u(x))

    u_x = np.array([2 - 4, 4 - 6, 6 - 8, 8 - 10, 10 - 12])
    assert(np.allclose(u(x), u_x))

    # visualization
    X, Y = np.mgrid[0:10:100j, 0:10:100j]
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    print(xy.shape)

    phi = np.array([10, 2])
    u = potentials.linear_DD(phi)
    z = u(xy).reshape(100, 100)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    ax[0].contourf(X, Y, z)
    ax[1].contourf(X, Y, 10 * X + 2 * Y)
    fig.savefig("test_potentials_outputs/linear_2D_test.png")

    assert(np.allclose(z, 10 * X + 2 * Y))


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
