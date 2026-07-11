# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tier-0+ checks for the KnotJob validator (Khovanov F2/Q, Rasmussen s).

Real ``knotjob`` subprocess runs here; the file skips only if the binary is not on
PATH. The convention claims these lock (verified empirically before wiring): KnotJob
reads Tetradrome's PD in the mirror convention, so (h, q) -> (-h, -q) and s -> -s land
on native's values exactly. The sweep includes chiral knots (nonzero s), so a missing
mirror or sign flip cannot pass. One run yields all three invariants: the rational
theory is the integral free part, F2 adds the order-2 torsion via UCT.
"""
import shutil

import pytest

if shutil.which("knotjob") is None:
    pytest.skip("knotjob binary not on PATH", allow_module_level=True)

from tetradrome import invariants, knots
from tetradrome.backends.knotjob_adapter import (
    KnotJobValidator,
    _f2_from_integral,
    _parse_khovanov,
)

COVERED = ["khovanov_homology", "rational_khovanov_homology", "rasmussen_s"]
SWEEP = ["3_1", "4_1", "5_2", "8_19"]


def test_validator_available_and_versioned():
    validator = KnotJobValidator()
    assert validator.is_available() is True
    info = validator.version_info()
    assert set(info) == {"knotjob"}
    assert info["knotjob"].startswith("sha256:")


@pytest.mark.parametrize("name", SWEEP)
@pytest.mark.parametrize("invariant", COVERED)
def test_knotjob_matches_native_canonical(name, invariant):
    """KnotJob's value under the canonical name and convention equals the native one."""
    knot = knots.from_name(name)
    native = invariants.compute(knot, invariant, validate="off").value
    assert KnotJobValidator().known_value(knot, invariant) == native


def test_anchor_values_for_chiral_knots():
    """Hard-coded canonical anchors so validator and native cannot drift together.
    s of KnotInfo's PD is the mirror of its tabulated value: left trefoil -2, T(3,4) -6."""
    validator = KnotJobValidator()
    assert validator.known_value(knots.from_name("3_1"), "rasmussen_s") == -2
    assert validator.known_value(knots.from_name("8_19"), "rasmussen_s") == -6
    ranks = validator.known_value(knots.from_name("3_1"), "rational_khovanov_homology")
    assert sum(ranks.values()) == 4  # unreduced trefoil: rank 4 over Q
    assert min(h for h, _ in ranks) == -3  # left-handed: homological degrees in [-3, 0]


def test_uncovered_invariant_and_pdless_knot_return_none():
    validator = KnotJobValidator()
    assert validator.known_value(knots.from_name("3_1"), "determinant") is None
    braid_only = knots.from_braid([1, 1, 1])  # no PD diagram
    assert validator.known_value(braid_only, "khovanov_homology") is None


def test_parse_khovanov_and_uct():
    assert _parse_khovanov("q^-1 + q + t^2q^5 + t^3q^9") == {
        (0, -1): 1, (0, 1): 1, (2, 5): 1, (3, 9): 1,
    }
    with pytest.raises(ValueError):
        _parse_khovanov("")
    # A Z/2 summand at (h, q) contributes F2 classes at (h, q) and (h-1, q).
    assert _f2_from_integral({(0, 1): 1}, {(3, 9): 1}) == {
        (0, 1): 1, (3, 9): 1, (2, 9): 1,
    }


def test_end_to_end_strict_records_knotjob():
    result = invariants.compute(knots.from_name("4_1"), "rasmussen_s")  # strict
    record = next(v for v in result.validation.validators if v.oracle == "knotjob")
    assert record.verdict == "pass"
    assert record.version.startswith("knotjob sha256:")
