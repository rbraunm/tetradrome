"""Tests for the grid differential and its homology (engine Phase 6).

Structural: the differential lowers Maslov by one, preserves Alexander, and squares to zero.
Against KnotInfo: HFK-hat matches the tabulated ranks and the top Alexander grading recovers
the three-genus, by direct equality over the derived roster (n <= 8 by default, n <= 10 under
--heavy). The KnotInfo grid_notation chirality has matched the tabulated convention across the
validated set; a mirror mismatch on the wider sweep would be a chirality (D1) finding to
systematize, not something to paper over with an up-to-mirror fallback.
"""
from collections import defaultdict

import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.floer import differential, hfk_hat, seifert_genus
from tetradrome.engines.floer.grid import GridDiagram
from tetradrome.engines.floer.gradings import alexander, maslov

STRUCTURAL_KNOTS = ["3_1", "4_1"]          # the d^2 sweep over all n! states, kept cheap



@pytest.mark.parametrize("name", STRUCTURAL_KNOTS)
def test_differential_grades_and_squares_to_zero(name):
    grid = GridDiagram.from_knotinfo(name)
    for state in grid.generators():
        m, a = maslov(grid, state), alexander(grid, state)
        composite = defaultdict(int)
        for y in differential(grid, state):
            assert maslov(grid, y) == m - 1
            assert alexander(grid, y) == a
            for z in differential(grid, y):
                composite[z] += 1
        assert all(count % 2 == 0 for count in composite.values())


def test_hfk_matches_knotinfo(floer_knot):
    name, _ = floer_knot
    grid = GridDiagram.from_knotinfo(name)
    assert hfk_hat(grid) == ki.hfk_ranks(name)


def test_genus_matches_three_genus(floer_knot):
    name, _ = floer_knot
    grid = GridDiagram.from_knotinfo(name)
    assert seifert_genus(grid) == int(ki.lookup(name)["three_genus"])
