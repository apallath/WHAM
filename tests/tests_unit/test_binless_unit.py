"""Unit tests for WHAM.binless"""
import inspect
import os
import pickle
import re
import sys

import numpy as np
import pytest

import WHAM.binless


@pytest.fixture(autouse=True)
def arrange():
    """Prepares temp directories for test output."""
    if not os.path.exists("test_out"):
        os.makedirs("test_out")


def test_NLL():
    pass


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
