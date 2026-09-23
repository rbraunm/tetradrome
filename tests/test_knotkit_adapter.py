# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tier-0+ checks for the knotkit validator (Rasmussen s over Q).

Real ``kk`` subprocess runs here; the file skips only if the binary is not on PATH. The
convention these lock (verified on the chiral sweep before wiring): fed our PD, knotkit's
s is the negation of canonical, so s -> -s lands on native exactly while the raw value
does not. The sweep includes chiral knots, so a missed transform cannot pass.
"""
import shutil

import pytest

if shutil.which("kk") is None:
    pytest.skip("kk (knotkit) binary not on PATH", allow_module_level=True)

from tetradrome import invariants, knots
from tetradrome.backends.knotkit_adapter import KnotkitValidator, parse_s, raw_knotkit_s

SWEEP = ["3_1", "4_1", "5_2", "8_19", "10_124"]


def test_validator_available_and_versioned():
    validator = KnotkitValidator()
    assert validator.name == "knotkit"
    assert validator.covered_invariants == {"rasmussen_s"}
    assert validator.is_available() is True
    version = validator.version_info()
    assert set(version) == {"knotkit"}
    assert version["knotkit"].startswith("git:")


@pytest.mark.parametrize("name", SWEEP)
def test_knotkit_s_matches_native(name):
    knot = knots.from_name(name)
    native = invariants.compute(knot, "rasmussen_s", validate="off").value
    assert KnotkitValidator().known_value(knot, "rasmussen_s") == native


@pytest.mark.parametrize("name", ["3_1", "5_2", "8_19", "10_124"])
def test_raw_value_is_the_negation_on_chiral_knots(name):
    """The wiring claim itself: on every chiral knot the raw kk value is wrong and its
    negation is right, so the adapter cannot be passing on an identity transform."""
    knot = knots.from_name(name)
    native = invariants.compute(knot, "rasmussen_s", validate="off").value
    raw = parse_s(raw_knotkit_s(knot))
    assert native != 0
    assert raw == -native


def test_anchor_values():
    validator = KnotkitValidator()
    assert validator.known_value(knots.from_name("3_1"), "rasmussen_s") == -2
    assert validator.known_value(knots.from_name("4_1"), "rasmussen_s") == 0
    assert validator.known_value(knots.from_name("10_124"), "rasmussen_s") == -8


def test_uncovered_invariant_returns_none():
    knot = knots.from_name("3_1")
    assert KnotkitValidator().known_value(knot, "khovanov_homology") is None
    assert KnotkitValidator().known_value(knot, "jones_polynomial") is None


def test_parse_s_reads_exactly_one_line():
    assert parse_s("s(PD[X[1,5,2,4]]; Q) = 2\n") == 2
    for text in ("", "s(3_1; Q) = 2\ns(3_1; Q) = 2\n", "no s here\n"):
        with pytest.raises(ValueError, match="expected one s line"):
            parse_s(text)


def test_end_to_end_strict_records_knotkit_beside_knotjob():
    """Strict consults every wired s validator; knotkit makes it two independent ones."""
    result = invariants.compute(knots.from_name("5_2"), "rasmussen_s")
    verdicts = {record.oracle: record.verdict for record in result.validation.validators}
    assert verdicts["knotkit"] == "pass"
    assert verdicts["knotjob"] == "pass"
    record = next(v for v in result.validation.validators if v.oracle == "knotkit")
    assert record.version.startswith("knotkit git:")
