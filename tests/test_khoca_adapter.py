# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tier-0+ checks for the Khoca validator (unreduced Khovanov over F2 and Q).

Real ``khoca`` calls here; the file skips only if the module is not importable. The
convention claim these lock (verified empirically on the chiral sweep before wiring):
khoca's quantum grading is negated relative to canonical, so (h, q) -> (h, -q) lands on
native's values exactly, while direct and full mirror do not. The sweep includes chiral
knots, so a missed transform cannot pass. Each field ring is computed natively, so F2 is
not a UCT derivation from an integral run -- unlike the KnotJob path.
"""
import pytest

khoca = pytest.importorskip("khoca", reason="khoca module not installed")

from tetradrome import invariants, knots
from tetradrome.backends.khoca_adapter import (
    KhocaValidator,
    _unreduced_groups,
    raw_khoca,
)

COVERED = ["khovanov_homology", "rational_khovanov_homology"]
SWEEP = ["3_1", "4_1", "5_2", "8_19"]


def test_validator_available_and_versioned():
    validator = KhocaValidator()
    assert validator.name == "khoca"
    assert validator.covered_invariants == set(COVERED)
    assert validator.is_available() is True
    version = validator.version_info()
    assert set(version) == {"khoca"}
    assert version["khoca"] != "absent"


@pytest.mark.parametrize("name", SWEEP)
@pytest.mark.parametrize("invariant", COVERED)
def test_khoca_matches_native_canonical(name, invariant):
    knot = knots.from_name(name)
    native = invariants.compute(knot, invariant, validate="off").value
    assert KhocaValidator().known_value(knot, invariant) == native


def test_anchor_values_for_a_chiral_knot():
    """Right-handed trefoil, both rings, pinned exactly. F2 carries two more classes
    than Q -- the order-2 torsion showing up natively rather than through UCT."""
    knot = knots.from_name("3_1")
    validator = KhocaValidator()
    rational = validator.known_value(knot, "rational_khovanov_homology")
    f2 = validator.known_value(knot, "khovanov_homology")
    assert rational == {(0, -1): 1, (0, -3): 1, (-2, -5): 1, (-3, -9): 1}
    assert f2 == {
        (0, -1): 1, (0, -3): 1,
        (-2, -5): 1, (-2, -7): 1,
        (-3, -7): 1, (-3, -9): 1,
    }
    assert sum(f2.values()) == sum(rational.values()) + 2


def test_q_negation_is_the_only_transform_that_lands():
    """The wiring claim itself: direct and full mirror must both fail, or the adapter
    would be applying a transform that merely happens to work on one symmetric knot."""
    knot = knots.from_name("3_1")
    native = invariants.compute(knot, "rational_khovanov_homology", validate="off").value
    rows = raw_khoca(knot, 1)[1]
    direct = {}
    mirror = {}
    for t, q, _torsion, multiplicity in rows:
        direct[(t, q)] = direct.get((t, q), 0) + multiplicity
        mirror[(-t, -q)] = mirror.get((-t, -q), 0) + multiplicity
    assert direct != native
    assert mirror != native


def test_uncovered_invariant_and_pdless_knot_return_none():
    validator = KhocaValidator()
    knot = knots.from_name("3_1")
    assert validator.known_value(knot, "jones_polynomial") is None
    assert validator.known_value(knot, "rasmussen_s") is None


def test_torsion_row_from_a_field_ring_fails_loud():
    """A field coefficient ring cannot produce torsion; if khoca ever emits one the
    adapter must raise rather than silently dropping the row."""
    with pytest.raises(ValueError, match="torsion row"):
        _unreduced_groups([[], [[0, -1, 2, 1]]], 2)


def test_negative_aggregate_rank_fails_loud():
    with pytest.raises(ValueError, match="negative aggregate rank"):
        _unreduced_groups([[], [[0, -1, 0, 1], [0, -1, 0, -3]]], 1)


def test_empty_homology_fails_loud():
    with pytest.raises(ValueError, match="empty Khovanov homology"):
        _unreduced_groups([[], []], 1)


def test_end_to_end_strict_records_khoca():
    """Strict must consult khoca and record a passing verdict on the result."""
    result = invariants.compute(knots.from_name("5_2"), "rational_khovanov_homology")
    record = next(v for v in result.validation.validators if v.oracle == "khoca")
    assert record.verdict == "pass"
    assert record.version.startswith("khoca ")
