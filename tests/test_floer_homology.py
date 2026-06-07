"""Tests for the grid differential and its homology (engine Phase 6).

Structural: the differential lowers Maslov by one, preserves Alexander, and squares to zero.
Against KnotInfo: the resulting HFK-hat matches the tabulated ranks up to mirror (the flat
marker list does not fix chirality), and the top Alexander grading recovers the three-genus.
"""
from collections import defaultdict

import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.floer import differential, hfk_hat, seifert_genus
from tetradrome.engines.floer.grid import GridDiagram
from tetradrome.engines.floer.gradings import alexander, maslov

HFK_KNOTS = ["3_1", "4_1", "5_1", "5_2"]   # n up to 7
STRUCTURAL_KNOTS = ["3_1", "4_1"]          # the d^2 sweep over all n! states, kept cheap


def _mirror(ranks):
    return {(-m, -a): c for (m, a), c in ranks.items()}


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


@pytest.mark.parametrize("name", HFK_KNOTS)
def test_hfk_matches_knotinfo_up_to_mirror(name):
    grid = GridDiagram.from_knotinfo(name)
    computed = hfk_hat(grid)
    oracle = ki.hfk_ranks(name)
    assert computed == oracle or computed == _mirror(oracle)


@pytest.mark.parametrize("name", HFK_KNOTS)
def test_genus_matches_three_genus(name):
    grid = GridDiagram.from_knotinfo(name)
    assert seifert_genus(grid) == int(ki.lookup(name)["three_genus"])
