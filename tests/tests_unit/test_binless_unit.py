"""Unit tests for WHAM.binless"""
import inspect
import os
import pickle
import re
import sys

import autograd
import numpy as np
import pytest

import WHAM.binless


@pytest.fixture(autouse=True)
def arrange():
    """Prepares temp directories for test output."""
    if not os.path.exists("test_out"):
        os.makedirs("test_out")


def test_NLL():
    """Compares analytical gradient against autograd gradient."""
    g_i = np.random.rand(20)
    N_i = np.random.randint(low=1, high=10, size=20)
    Ntot = N_i.sum()
    W_il = np.random.rand(20, Ntot)

    print(g_i.shape, N_i.shape, W_il.shape)

    # Compare NLL with autograd=True and autograd=False
    assert np.isclose(WHAM.binless.CalcBase.NLL(g_i, N_i, W_il, autograd=True), WHAM.binless.CalcBase.NLL(g_i, N_i, W_il, autograd=False))

    # autograd.value_and_grad()
    print(autograd.grad(WHAM.binless.CalcBase.NLL, 0)(g_i, N_i, W_il, autograd=True))
    print(WHAM.binless.CalcBase.grad_NLL(g_i, N_i, W_il))
    assert np.allclose(autograd.grad(WHAM.binless.CalcBase.NLL, 0)(g_i, N_i, W_il, autograd=True), WHAM.binless.CalcBase.grad_NLL(g_i, N_i, W_il))

################################################################################
# Serialization tests
################################################################################


def test_serialization_1D():
    calc = WHAM.binless.Calc1D()
    x_l = np.random.random(100)
    G_l = np.random.random(100)
    g_i = np.random.random(5)
    calc.x_l = x_l
    calc.G_l = G_l
    calc.g_i = g_i

    with open("test_out/calc1D.pkl", "wb") as calcf:
        pickle.dump(calc, calcf)

    with open("test_out/calc1D.pkl", "rb") as calcf:
        calc_load = pickle.load(calcf)

    assert(np.allclose(calc_load.x_l, calc.x_l))
    assert(np.allclose(calc_load.G_l, calc.G_l))
    assert(np.allclose(calc_load.g_i, calc.g_i))


def test_serialization_DD():
    calc = WHAM.binless.CalcDD()
    x_l = np.random.random((100, 10))
    G_l = np.random.random(100)
    g_i = np.random.random(5)
    calc.x_l = x_l
    calc.G_l = G_l
    calc.g_i = g_i

    with open("test_out/calcDD.pkl", "wb") as calcf:
        pickle.dump(calc, calcf)

    with open("test_out/calcDD.pkl", "rb") as calcf:
        calc_load = pickle.load(calcf)

    assert(np.allclose(calc_load.x_l, calc.x_l))
    assert(np.allclose(calc_load.G_l, calc.G_l))
    assert(np.allclose(calc_load.g_i, calc.g_i))


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
