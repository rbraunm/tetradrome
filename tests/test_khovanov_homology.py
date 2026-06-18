# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Validate native F2 Khovanov homology against KnotInfo (Phase 2c).

The expected value comes from knotinfo_backend.known_answer, which derives the mod-2
Betti numbers from KnotInfo's stored integral vector by universal coefficients and
mirrors them to our chirality (Phase 2c). That derivation and our cube computation are
fully independent code paths, so their agreement is genuine validation.
"""
import pytest

from tetradrome import knots
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines import khovanov

# <= 7 crossings: enhanced-state enumeration is exponential.
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]


@pytest.mark.parametrize("name", KNOTS)
def test_khovanov_matches_knotinfo(name):
    pd = knots.from_name(name).pd_code
    assert khovanov.khovanov_homology(pd) == ki.known_answer(name, "khovanov_homology")


def test_unknot_khovanov():
    # Unreduced Khovanov of the unknot over F2: F2 in (0, +1) and (0, -1).
    assert khovanov.khovanov_homology(()) == {(0, 1): 1, (0, -1): 1}


def test_trefoil_explicit():
    # KnotInfo's 3_1 PD is the left-handed trefoil; its F2 Khovanov (Knot Atlas).
    pd = knots.from_name("3_1").pd_code
    assert khovanov.khovanov_homology(pd) == {
        (0, -1): 1,
        (0, -3): 1,
        (-2, -5): 1,
        (-2, -7): 1,
        (-3, -7): 1,
        (-3, -9): 1,
    }
