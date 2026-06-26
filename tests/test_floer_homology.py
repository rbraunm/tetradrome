# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the grid differential and its homology (engine Phase 6).

Structural: the differential lowers Maslov by one, preserves Alexander, and squares to zero.
Against KnotInfo: HFK-hat matches the tabulated ranks and its top Alexander grading recovers
the three-genus, by direct equality over the derived roster (n <= 8 serial by default, n <= 10
across all cores under --heavy). HFK and the genus share one computation per knot. The KnotInfo
grid_notation chirality has matched the tabulated convention across the validated set; a mirror
mismatch on the wider sweep would be a chirality (D1) finding to systematize, not something to
paper over with an up-to-mirror fallback.
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


def test_hfk_and_genus_match_knotinfo(floer_knot):
    name, _ = floer_knot
    grid = GridDiagram.from_knotinfo(name)
    ranks = hfk_hat(grid)        # one computation; genus reads its top grading
    assert ranks == ki.hfk_ranks(name)
    assert max(a for _m, a in ranks) == int(ki.lookup(name)["three_genus"])


def test_seifert_genus_reads_top_grading():
    # cover the public wrapper directly (the roster test derives the genus inline)
    assert seifert_genus(GridDiagram.from_knotinfo("4_1")) == 1
