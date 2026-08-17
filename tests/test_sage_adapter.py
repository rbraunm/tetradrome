# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The gated SageValidator: unusable-by-construction until CT 250 verifies it.

Everything here runs WITHOUT sage -- these tests pin the gate itself and the
single-source contract between the validator and its verification probe. The
sage-required behavior tests (verified transforms against native, end-to-end strict
for signature) land at wiring time, once _VERIFIED_VERDICTS is transcribed from a
CONSISTENT CT 250 probe run.

TODO (wiring time, CT 250): known_value equals native across the chiral sweep per
verified verdict; compute(signature) validates under strict with a sage record.
"""
import shutil

import pytest

from tetradrome import knots
from tetradrome.backends import registry, sage_adapter
from tetradrome.backends.sage_adapter import SageValidator


def test_unverified_gate_raises_before_anything_else():
    validator = SageValidator()
    knot = knots.from_name("3_1")
    with pytest.raises(RuntimeError, match="unverified"):
        validator.known_value(knot, "signature")
    # Gate-first ordering: even inputs the validator could never check raise, so an
    # unverified validator cannot quietly look usable on any path.
    with pytest.raises(RuntimeError, match="unverified"):
        validator.known_value(knots.from_braid([1, 1, 1]), "signature")
    with pytest.raises(RuntimeError, match="unverified"):
        validator.known_value(knot, "three_genus")


def test_sage_is_not_wired_and_still_listed_unwired():
    assert "sage" not in [validator.name for validator in registry._WIRED]
    for invariant in ("signature", "determinant", "alexander_polynomial",
                      "jones_polynomial"):
        assert "sage" in registry.unwired_oracles(invariant)


def test_verdicts_start_unverified():
    assert sage_adapter._VERIFIED_VERDICTS is None


def test_transform_table_covers_every_covered_invariant():
    for invariant in sage_adapter._COVERED:
        labels = sage_adapter.candidate_labels(invariant)
        assert labels, f"{invariant} has no candidate transforms"
        assert invariant in sage_adapter._NEEDED_TAGS
        for tag in sage_adapter._NEEDED_TAGS[invariant]:
            assert tag in sage_adapter._TAG_LINES
    for (invariant, _label) in sage_adapter._TRANSFORMS:
        assert invariant in sage_adapter._COVERED


def test_contract_shape():
    validator = SageValidator()
    assert validator.name == "sage"
    assert validator.covered_invariants == sage_adapter._COVERED
    assert "signature" in validator.covered_invariants


@pytest.mark.skipif(shutil.which("sage") is not None,
                    reason="asserts the sage-less environment's version string")
def test_version_info_reports_absent_without_sage():
    assert SageValidator().version_info() == {"sage": "absent"}
