# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

import pytest

from tetradrome import invariants, knots
from tetradrome.invariants import seifert


def test_collins_matrix_trefoil():
    m = seifert.seifert_matrix_from_braid([1, 1, 1])
    assert seifert.determinant(m) == 3
    assert seifert.signature(m) == -2


def test_collins_matrix_figure_eight():
    m = seifert.seifert_matrix_from_braid([1, -2, 1, -2])
    assert seifert.determinant(m) == 5
    assert seifert.signature(m) == 0


def test_collins_rejects_zero_entry():
    with pytest.raises(ValueError):
        seifert.seifert_matrix_from_braid([1, 0, 1])


# A spread including non-alternating (8_19 is the first) and torus knots, where the
# braid-induced surface is non-minimal-genus -- the derived invariants must still
# match the oracle.
SWEEP = [
    "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3",
    "7_1", "7_3", "7_4", "8_19", "9_42", "10_124", "10_139",
]


@pytest.mark.parametrize("name", SWEEP)
def test_compute_validates_against_knotinfo(name):
    k = knots.from_name(name)
    # determinant is strict (regina cross-checks); signature stays soft off CT 250
    # (sage, its only computed oracle, is installed there alone).
    for inv, mode in (("determinant", "strict"), ("signature", "soft")):
        result = invariants.compute(k, inv, validate=mode)
        assert result.validation.verdict("knotinfo") == "pass"
