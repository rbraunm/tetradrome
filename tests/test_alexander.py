# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Alexander polynomial: det(V - t*V^T) from the Collins matrix, canonicalized.

Coefficients are ascending (constant term first), in the canonical form KnotInfo's
alexander_polynomial_vector uses (lowest power shifted to t^0, constant term positive).
"""
import pytest

from tetradrome import invariants, knots
from tetradrome.invariants import seifert


def test_alexander_trefoil_and_figure_eight():
    # 3_1: 1 - t + t^2 ; 4_1: 1 - 3t + t^2
    assert seifert.alexander_polynomial(seifert.seifert_matrix_from_braid([1, 1, 1])) == (1, -1, 1)
    assert seifert.alexander_polynomial(
        seifert.seifert_matrix_from_braid([1, -2, 1, -2])
    ) == (1, -3, 1)


def test_canonicalization_handles_unit_and_sign():
    # t^2*(1 - t + t^2) with a flipped sign must reduce to the same canonical form.
    assert seifert.canonical_alexander([0, 0, -1, 1, -1]) == (1, -1, 1)


def test_alexander_evaluated_at_minus_one_is_determinant():
    # |Delta(-1)| == knot determinant, for a few knots.
    for name, det in [("3_1", 3), ("4_1", 5), ("5_2", 7), ("6_1", 9)]:
        coeffs = invariants.compute(knots.from_name(name), "alexander_polynomial").value
        val = sum(c * (-1) ** i for i, c in enumerate(coeffs))
        assert abs(val) == det


SWEEP = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4", "8_19", "9_42", "10_124"]


@pytest.mark.parametrize("name", SWEEP)
def test_alexander_validates_against_knotinfo(name):
    result = invariants.compute(knots.from_name(name), "alexander_polynomial")
    assert result.validation.verdict("knotinfo") == "pass"


def test_conway_knot_alexander_is_trivial():
    # The Conway knot 11n34 has Alexander polynomial 1 -- this is exactly why it is
    # hard: the Alexander polynomial (and determinant) cannot see it.
    result = invariants.compute(knots.from_name("K11n34"), "alexander_polynomial")
    assert result.value == (1,)
    assert result.validation.verdict("knotinfo") == "pass"


def test_alexander_offtable_torus():
    # T(2,7) closed form is 1 - t + t^2 - ... + t^6 (alternating), here as an off-table
    # braid (validate=False since there is no oracle for a braid-only knot).
    coeffs = invariants.compute(
        knots.from_braid([1] * 7), "alexander_polynomial", validate=False
    ).value
    assert coeffs == (1, -1, 1, -1, 1, -1, 1)
