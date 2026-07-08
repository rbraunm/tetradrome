# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

import pytest

from tetradrome import invariants, knots
from tetradrome.errors import UnknownKnot
from tetradrome.invariants import jones


def test_trefoil_bracket():
    # <3_1> = -A^-5 - A^3 + A^7 under the A-smoothing convention used here.
    pd = knots.from_name("3_1").pd_code
    assert jones.kauffman_bracket(pd) == {-5: -1, 3: -1, 7: 1}


def test_jones_small_knots():
    cases = {
        "3_1": (1, (1, 0, 1, -1)),         # t + t^3 - t^4
        "4_1": (-2, (1, -1, 1, -1, 1)),    # t^-2 - t^-1 + 1 - t + t^2
        "5_2": (1, (1, -1, 2, -1, 1, -1)),
        "8_19": (3, (1, 0, 1, 0, 0, -1)),  # t^3 + t^5 - t^8
    }
    for name, expected in cases.items():
        assert jones.jones_polynomial(knots.from_name(name).pd_code) == expected


def test_unknot_jones():
    assert jones.jones_polynomial(()) == (0, (1,))


def test_amphichiral_jones_is_palindromic():
    # 4_1 is amphichiral, so its Jones polynomial is symmetric under t <-> 1/t.
    low, coeffs = jones.jones_polynomial(knots.from_name("4_1").pd_code)
    assert coeffs == tuple(reversed(coeffs))
    assert low == -(low + len(coeffs) - 1)


SWEEP = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4", "8_19", "9_42", "10_124"]


@pytest.mark.parametrize("name", SWEEP)
def test_jones_validates_against_knotinfo(name):
    result = invariants.compute(knots.from_name(name), "jones_polynomial")
    assert result.validation.verdict("knotinfo") == "pass"


def test_jones_via_compute_value_and_provenance():
    result = invariants.compute(knots.from_name("3_1"), "jones_polynomial")
    assert result.value == (1, (1, 0, 1, -1))
    assert result.provenance.method == "kauffman_bracket"
    assert result.provenance.inputs == "pd_code"


def test_jones_needs_a_pd_diagram():
    k = knots.from_braid([1, 1, 1])  # braid-only, no PD
    with pytest.raises(UnknownKnot):
        invariants.compute(k, "jones_polynomial", validate=False)
