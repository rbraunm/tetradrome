# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tier-0+ checks for the Regina validator (determinant, Alexander, Jones).

Real regina runs here; the file skips only if regina is not installed. The convention
claims these lock (verified empirically before wiring): regina's Jones in x = t^(1/2)
halves to native's t DIRECTLY, its Alexander canonicalizes to native's exact form, and
determinant is |Delta(-1)| through regina's independent code path. The sweep includes
chiral knots, so a stray t <-> t^-1 flip or sign error cannot pass.
"""
import pytest

pytest.importorskip("regina")

from tetradrome import invariants, knots
from tetradrome.backends.regina_adapter import ReginaValidator, _parse_laurent

COVERED = ["determinant", "alexander_polynomial", "jones_polynomial"]
SWEEP = ["3_1", "4_1", "5_2", "8_19", "10_124"]


def test_validator_available_and_versioned():
    validator = ReginaValidator()
    assert validator.is_available() is True
    info = validator.version_info()
    assert set(info) == {"regina"}
    assert info["regina"] not in ("", "absent")


@pytest.mark.parametrize("name", SWEEP)
@pytest.mark.parametrize("invariant", COVERED)
def test_regina_matches_native_canonical(name, invariant):
    """Regina's value under the canonical name and convention equals the native one."""
    knot = knots.from_name(name)
    native = invariants.compute(knot, invariant, validate="off").value
    assert ReginaValidator().known_value(knot, invariant) == native


def test_anchor_values_for_the_trefoil():
    """Hard-coded canonical anchors so validator and native cannot drift together."""
    knot = knots.from_name("3_1")
    validator = ReginaValidator()
    assert validator.known_value(knot, "determinant") == 3
    assert validator.known_value(knot, "alexander_polynomial") == (1, -1, 1)
    assert validator.known_value(knot, "jones_polynomial") == (1, (1, 0, 1, -1))


def test_uncovered_invariant_and_pdless_knot_return_none():
    validator = ReginaValidator()
    assert validator.known_value(knots.from_name("3_1"), "signature") is None
    braid_only = knots.from_braid([1, 1, 1])  # no PD diagram
    assert validator.known_value(braid_only, "determinant") is None


def test_parse_laurent_handles_regina_forms():
    assert _parse_laurent("-x^8 + x^6 + x^2", "x") == {8: -1, 6: 1, 2: 1}
    assert _parse_laurent("x^2 - x + 1", "x") == {2: 1, 1: -1, 0: 1}
    assert _parse_laurent("2x^-2 - 3 + 2x^2", "x") == {-2: 2, 0: -3, 2: 2}


def test_parse_laurent_fails_loud_on_junk():
    with pytest.raises(ValueError):
        _parse_laurent("", "x")
    with pytest.raises(ValueError):
        _parse_laurent("0", "x")
    with pytest.raises(Exception):
        _parse_laurent("x^2 + banana", "x")


def test_end_to_end_strict_records_regina():
    result = invariants.compute(knots.from_name("5_2"), "jones_polynomial")  # strict
    record = next(v for v in result.validation.validators if v.oracle == "regina")
    assert record.verdict == "pass"
    assert record.version.startswith("regina ")
