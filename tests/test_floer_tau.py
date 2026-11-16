# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the grid tau invariant (engine Phase 6).

tau matches KnotInfo's Ozsvath-Szabo tau directly, including the sign, over the derived roster
(n <= 8 by default, n <= 10 under --heavy); the roster includes 8_19 (the (3,4)-torus knot,
tau = 3), a value well above the small alternating knots. A separate test pins that the sign
tracks chirality (mirror negates tau). ``tau`` runs through the scheduler; it is pinned against
the serial ``tau_reference`` and, piece by piece, against the reference's own matrices.
"""
from collections import defaultdict

import pytest

from tetradrome.algebra import f2_rank, pack_columns
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.floer import filtered_differential, maslov, tau
from tetradrome.engines.floer.generation import grading_histogram_reference
from tetradrome.engines.floer.grid import GridDiagram
from tetradrome.engines.floer.scheduling import tau_graph
from tetradrome.engines.floer.tau import tau_reference
from tetradrome.scheduler import Scheduler, detect_machine


def test_tau_matches_knotinfo(floer_knot, floer_tau):
    name, _ = floer_knot
    values, failures = floer_tau
    assert name not in failures, failures.get(name)
    assert values[name] == ki.tau_invariant(name)


@pytest.mark.parametrize("name", ["3_1", "4_1", "5_1", "5_2", "6_1", "8_19"])
def test_scheduled_tau_matches_serial_reference(name):
    grid = GridDiagram.from_knotinfo(name)
    assert tau(grid) == tau_reference(grid)


def test_tau_graph_reports_the_reference_ranks():
    # The graph's result carries the Maslov-0 generator count and the two boundary ranks beside
    # the level, so the merged matrices are pinned against the reference's, not only the final
    # answer. slice_states is forced small: the trefoil's 120 states split into eight slices.
    grid = GridDiagram.from_knotinfo("3_1")
    by_maslov = defaultdict(list)
    for state in grid.generators():
        by_maslov[maslov(grid, state)].append(state)
    c0, c_minus, c1 = by_maslov[0], by_maslov[-1], by_maslov[1]
    minus_index = {state: i for i, state in enumerate(c_minus)}
    zero_index = {state: i for i, state in enumerate(c0)}
    d0 = [frozenset(minus_index[y] for y in filtered_differential(grid, s)) for s in c0]
    d1 = [frozenset(zero_index[y] for y in filtered_differential(grid, s)) for s in c1]

    graph, assemble_key = tau_graph(grid, grading_histogram_reference(grid), slice_states=16)
    report = Scheduler(detect_machine()).run(graph)
    assert report.failures == []
    assert report.results[assemble_key] == (
        len(c0), f2_rank(pack_columns(d0)), f2_rank(pack_columns(d1)), tau_reference(grid))


def test_tau_sign_tracks_chirality():
    # The right- and left-handed trefoils have opposite tau; the grid distinguishes them.
    trefoil = GridDiagram.from_knotinfo("3_1")
    mirror = GridDiagram(list(reversed(trefoil.O)), list(reversed(trefoil.X)))
    assert tau(trefoil) == -tau(mirror)
    assert tau(trefoil) != 0
