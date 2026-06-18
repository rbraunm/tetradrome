# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the Lee deformation (engines/khovanov/lee.py).

The validation is Lee's theorem: the Lee homology of a knot is exactly 2-dimensional
over Q. Computing it also runs verify_d_squared over Q, so a wrong deformed map or sign
would surface as d^2 != 0 (or a wrong dimension) rather than passing silently. For a
knot both surviving generators sit in homological degree 0.
"""
import pytest

from tetradrome import knots
from tetradrome.engines import khovanov

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]


@pytest.mark.parametrize("name", KNOTS)
def test_lee_homology_is_two_dimensional(name):
    pd = knots.from_name(name).pd_code
    assert khovanov.lee_homology(pd) == {0: 2}


@pytest.mark.parametrize("name", KNOTS)
def test_lee_complex_d_squared_holds(name):
    pd = knots.from_name(name).pd_code
    khovanov.lee_complex(pd).verify_d_squared()  # must not raise


@pytest.mark.parametrize("name", KNOTS)
def test_lee_complex_has_same_generators_as_khovanov(name):
    # The deformation changes the maps, not the chain groups.
    pd = knots.from_name(name).pd_code
    assert khovanov.lee_complex(pd).total_dim() == khovanov.unreduced_size(pd)


def test_unknot_lee_homology():
    assert khovanov.lee_homology(()) == {0: 2}


def test_empty_diagram_rejected_in_cube():
    with pytest.raises(ValueError, match=r"empty diagram"):
        khovanov.lee_complex(())
