"""Unit tests for WHAM.binned"""
import sys
import inspect
import re

import numpy as np

import WHAM.binned


def test_NLL():
    pass


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
