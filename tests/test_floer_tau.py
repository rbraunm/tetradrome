"""Tests for the grid tau invariant (engine Phase 6).

With the grid in the standard chirality, tau matches KnotInfo's Ozsvath-Szabo tau directly,
including the sign (which the mirror-resolved chirality fixes). 8_19 (the (3,4)-torus knot,
tau = 3) exercises a value well above the small alternating knots.
"""
import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.floer import tau
from tetradrome.engines.floer.grid import GridDiagram

TAU_KNOTS = ["3_1", "4_1", "5_1", "5_2", "8_19"]   # n up to 7


@pytest.mark.parametrize("name", TAU_KNOTS)
def test_tau_matches_knotinfo(name):
    grid = GridDiagram.from_knotinfo(name)
    assert tau(grid) == ki.tau_invariant(name)


def test_tau_sign_tracks_chirality():
    # The right- and left-handed trefoils have opposite tau; the grid distinguishes them.
    trefoil = GridDiagram.from_knotinfo("3_1")
    mirror = GridDiagram(list(reversed(trefoil.O)), list(reversed(trefoil.X)))
    assert tau(trefoil) == -tau(mirror)
    assert tau(trefoil) != 0
