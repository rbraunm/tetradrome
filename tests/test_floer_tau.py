# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the grid tau invariant (engine Phase 6).

tau matches KnotInfo's Ozsvath-Szabo tau directly, including the sign, over the derived roster
(n <= 8 by default, n <= 10 under --heavy); the roster includes 8_19 (the (3,4)-torus knot,
tau = 3), a value well above the small alternating knots. A separate test pins that the sign
tracks chirality (mirror negates tau).
"""
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.floer import tau
from tetradrome.engines.floer.grid import GridDiagram

def test_tau_matches_knotinfo(floer_knot):
    name, _ = floer_knot
    grid = GridDiagram.from_knotinfo(name)
    assert tau(grid) == ki.tau_invariant(name)


def test_tau_sign_tracks_chirality():
    # The right- and left-handed trefoils have opposite tau; the grid distinguishes them.
    trefoil = GridDiagram.from_knotinfo("3_1")
    mirror = GridDiagram(list(reversed(trefoil.O)), list(reversed(trefoil.X)))
    assert tau(trefoil) == -tau(mirror)
    assert tau(trefoil) != 0
