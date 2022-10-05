# -*- coding: utf-8 -*-

import pytest

from hpar_reader.skeleton import fib

__author__ = "mark.tupas@geo.tuwien.ac.at"
__copyright__ = "mark.tupas@geo.tuwien.ac.at"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
