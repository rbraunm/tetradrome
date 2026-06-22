# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for exact reduction by Gaussian cancellation (engine Phase 4).

The headline is `raw == reduced`: cancelling every unit in the differential is an exact
homotopy equivalence, so over a field the surviving generators must reproduce the
rank-counted homology -- an independent algorithm agreeing with the reference reducers
(and, transitively, with KnotInfo). Cancellation also shrinks the complex from its full
unreduced size down to the homology dimension, the basis for the engine's memory tool.
"""
from fractions import Fraction

import pytest

from tetradrome import knots
from tetradrome.algebra import (
    GradedComplex,
    RationalComplex,
    gaussian_homology,
    homology,
    rational_homology,
)
from tetradrome.engines import khovanov
from tetradrome.engines.khovanov.lee import lee_complex

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]
SINGLE_COMPLEX_KNOTS = ["3_1", "4_1", "5_2", "6_2", "7_4"]  # Lee is one big complex


def test_handbuilt_f2_matches_rank():
    # x2 map is zero mod 2 -> F2^2; a merge map -> rank 1.
    cx = GradedComplex({0: 2, 1: 1}, {0: [{0}, {0}]})
    assert gaussian_homology(cx) == homology(cx) == {0: 1}


def test_handbuilt_rational_matches_rank():
    # [[2,1],[1,3]] (det 5) is acyclic over Q; exercises Fraction inverses in cancellation.
    cx = RationalComplex({0: 2, 1: 2}, {0: [{0: 2, 1: 1}, {0: 1, 1: 3}]})
    assert gaussian_homology(cx) == rational_homology(cx) == {}
    cx2 = RationalComplex({0: 1, 1: 1}, {0: [{0: Fraction(2)}]})  # x2: iso over Q
    assert gaussian_homology(cx2) == {}


@pytest.mark.parametrize("name", KNOTS)
def test_f2_khovanov_cancellation_matches_rank(name):
    pd = knots.from_name(name).pd_code
    for cx in khovanov.khovanov_complexes(pd).values():
        assert gaussian_homology(cx) == homology(cx)


@pytest.mark.parametrize("name", KNOTS)
def test_rational_khovanov_cancellation_matches_rank(name):
    pd = knots.from_name(name).pd_code
    for cx in khovanov.khovanov_complexes_q(pd).values():
        assert gaussian_homology(cx) == rational_homology(cx)


@pytest.mark.parametrize("name", SINGLE_COMPLEX_KNOTS)
def test_lee_cancellation_matches_rank_and_shrinks(name):
    pd = knots.from_name(name).pd_code
    cx = lee_complex(pd)
    reduced = gaussian_homology(cx)
    assert reduced == rational_homology(cx) == {0: 2}
    # the whole point of the size tool: reduce the full cube down to the 2-dim homology.
    assert sum(reduced.values()) == 2 < cx.total_dim()


def test_bigraded_khovanov_via_cancellation_matches_reference():
    # Assemble the full F2 Khovanov table by cancellation; must equal the rank-based one.
    pd = knots.from_name("5_2").pd_code
    via_cancel = {}
    for j, cx in khovanov.khovanov_complexes(pd).items():
        for i, d in gaussian_homology(cx).items():
            via_cancel[(i, j)] = d
    assert via_cancel == khovanov.khovanov_homology(pd)


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        gaussian_homology(object())
