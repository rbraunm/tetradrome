# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The Sage validator: verified transforms, wired, CT 250-installed only.

The sandbox-runnable half pins the transcription (every verdict label resolves to a
real transform), the contract, and the sage-less host's honest states. The
sage-required half runs wherever sage is installed (CT 250): regression pins for the
verified transforms against native, and the end-to-end strict signature result the
whole sage arc existed to unlock.
"""
import shutil

import pytest

from tetradrome import invariants, knots
from tetradrome.backends import registry, sage_adapter
from tetradrome.backends.sage_adapter import SageValidator

SAGE_PRESENT = shutil.which("sage") is not None


# --- sandbox-runnable: transcription, contract, and sage-less states ---

def test_verdicts_are_transcribed_and_resolve_to_transforms():
    verdicts = sage_adapter._VERIFIED_VERDICTS
    assert set(verdicts) == sage_adapter._COVERED
    for invariant, label in verdicts.items():
        assert (invariant, label) in sage_adapter._TRANSFORMS
    # The CT 250 probe's verdicts, verbatim (2026-08-17, SageMath version 9.5).
    assert verdicts["signature"] == "NEGATED"
    assert verdicts["jones_polynomial"] == "NEGATED_EXPONENTS"
    assert verdicts["determinant"] == "DIRECT"
    assert verdicts["alexander_polynomial"] == "CANONICAL"
    assert verdicts["rational_khovanov_homology"] == "DIRECT"
    assert verdicts["khovanov_homology"] == "DIRECT"


def test_sage_is_wired_and_classical_unwired_entries_are_gone():
    assert "sage" in [validator.name for validator in registry._WIRED]
    for invariant in ("signature", "determinant", "alexander_polynomial",
                      "jones_polynomial"):
        assert registry.unwired_oracles(invariant) == ()
        assert registry.computed_oracle_exists(invariant)


def test_contract_shape():
    validator = SageValidator()
    assert validator.name == "sage"
    assert validator.covered_invariants == sage_adapter._COVERED
    assert "signature" in validator.covered_invariants


def test_uncovered_invariant_and_pdless_knot_return_none():
    validator = SageValidator()
    assert validator.known_value(knots.from_name("3_1"), "three_genus") is None
    braid_only = knots.from_braid([1, 1, 1])  # no PD diagram
    assert validator.known_value(braid_only, "signature") is None


def test_transform_table_covers_every_covered_invariant():
    for invariant in sage_adapter._COVERED:
        assert sage_adapter.candidate_labels(invariant)
        assert invariant in sage_adapter._NEEDED_TAGS
        for tag in sage_adapter._NEEDED_TAGS[invariant]:
            assert tag in sage_adapter._TAG_LINES
    for (invariant, _label) in sage_adapter._TRANSFORMS:
        assert invariant in sage_adapter._COVERED


@pytest.mark.skipif(SAGE_PRESENT, reason="asserts the sage-less environment's states")
def test_sageless_host_states():
    validator = SageValidator()
    assert validator.is_available() is False
    assert validator.version_info() == {"sage": "absent"}


# --- sage-required (CT 250): verified transforms against native, end to end ---

CLASSICAL = ["signature", "determinant", "alexander_polynomial", "jones_polynomial"]

pytestmark_sage = pytest.mark.skipif(not SAGE_PRESENT, reason="sage not installed")


@pytestmark_sage
@pytest.mark.parametrize("name", ["3_1", "4_1"])  # chiral + amphichiral
@pytest.mark.parametrize("invariant", CLASSICAL)
def test_sage_matches_native_canonical_classical(name, invariant):
    knot = knots.from_name(name)
    native = invariants.compute(knot, invariant, validate="off").value
    assert SageValidator().known_value(knot, invariant) == native


@pytestmark_sage
@pytest.mark.parametrize(
    "invariant", ["rational_khovanov_homology", "khovanov_homology"]
)
def test_sage_matches_native_canonical_khovanov(invariant):
    knot = knots.from_name("3_1")  # chiral: a missing mirror cannot pass
    native = invariants.compute(knot, invariant, validate="off").value
    assert SageValidator().known_value(knot, invariant) == native


@pytestmark_sage
def test_signature_validates_under_strict_with_sage():
    """The result the sage arc existed to unlock: signature under default strict."""
    result = invariants.compute(knots.from_name("3_1"), "signature")
    assert result.value == -2
    record = next(v for v in result.validation.validators if v.oracle == "sage")
    assert record.verdict == "pass"
    assert record.version.startswith("sage SageMath")
    assert result.validation.verdict("knotinfo") == "pass"
    assert result.validation.is_validated
