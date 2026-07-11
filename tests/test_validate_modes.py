# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Three-mode validate (ADR 0004) against the validator registry.

Strict requires a computed oracle wherever one exists anywhere; soft tolerates a
computed oracle's absence but never a disagreement; off skips validation while keeping
full provenance. The stub validator stands in for an external oracle at the registry
boundary -- the mode logic in _finalize runs for real against it.
"""
import logging

import pytest

from tetradrome import invariants, knots
from tetradrome.backends import registry
from tetradrome.errors import UnvalidatedResult

REGINA_COVERED = ["determinant", "alexander_polynomial", "jones_polynomial"]
KNOTJOB_COVERED = ["khovanov_homology", "rational_khovanov_homology", "rasmussen_s"]
STRICT_UNWIRED = ["signature"]


# --- regina-covered invariants pass strict with a computed oracle on record ---

@pytest.mark.parametrize("invariant", REGINA_COVERED)
def test_regina_covered_invariants_pass_under_strict(invariant):
    result = invariants.compute(knots.from_name("3_1"), invariant)  # strict default
    assert result.validation.verdict("regina") == "pass"
    assert result.validation.verdict("knotinfo") == "pass"
    assert result.validation.is_validated


@pytest.mark.parametrize("invariant", KNOTJOB_COVERED)
def test_knotjob_covered_invariants_pass_under_strict(invariant):
    result = invariants.compute(knots.from_name("3_1"), invariant)  # strict default
    assert result.validation.verdict("knotjob") == "pass"
    assert result.validation.verdict("knotinfo") == "pass"
    assert result.validation.is_validated


# --- invariants with no wired computed oracle still raise under strict ---

@pytest.mark.parametrize("invariant", STRICT_UNWIRED)
def test_unwired_invariants_raise_under_strict_naming_every_unwired_oracle(invariant):
    k = knots.from_name("3_1")
    with pytest.raises(UnvalidatedResult) as excinfo:
        invariants.compute(k, invariant)  # strict is the default
    message = str(excinfo.value)
    assert "not yet wired" in message
    for oracle in registry.unwired_oracles(invariant):
        assert oracle in message


# --- soft: KnotInfo carries the result, with an info message naming the gap ---

def test_soft_falls_back_to_knotinfo_with_info_message(caplog):
    k = knots.from_name("3_1")
    with caplog.at_level(logging.INFO, logger="tetradrome.invariants.compute"):
        result = invariants.compute(k, "signature", validate="soft")
    assert result.value == -2
    assert result.validation.verdict("knotinfo") == "pass"
    assert result.validation.is_validated
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "strict would have raised" in m and "not yet wired" in m for m in messages
    )


def test_soft_still_raises_offtable_with_nothing_to_validate():
    # Soft tolerates a computed oracle's absence, but something must still validate:
    # an off-table braid has no KnotInfo value either.
    k = knots.from_braid([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # T(2,15)
    with pytest.raises(UnvalidatedResult, match="KnotInfo has no value"):
        invariants.compute(k, "determinant", validate="soft")


# --- off: no validators consulted, full provenance kept ---

def test_off_skips_all_validators_but_keeps_full_provenance():
    result = invariants.compute(knots.from_name("3_1"), "determinant", validate="off")
    assert result.value == 3
    assert result.validation.validators == ()
    assert not result.validation.is_validated
    assert result.validation.verdict("knotinfo") == "not_run"
    assert result.provenance.backend == "tetradrome-native"
    assert result.provenance.method == "seifert_form_from_braid"
    assert dict(result.provenance.library_versions)["python"]


# --- the mode guard fails loud, including on the retired boolean form ---

@pytest.mark.parametrize("bad", [True, False, "STRICT", "on", None, 1])
def test_validate_mode_guard_rejects_non_modes(bad):
    with pytest.raises(ValueError, match="strict"):
        invariants.compute(knots.from_name("3_1"), "determinant", validate=bad)


# --- mode logic against a wired computed oracle (stub at the registry boundary) ---

class _StubValidator:
    """A registry-shaped double for an external oracle (SPEC 12.1 contract)."""

    name = "stub_oracle"
    covered_invariants = {"determinant"}

    def __init__(self, value=None, available=True):
        self._value = value
        self._available = available

    def is_available(self):
        return self._available

    def version_info(self):
        return {"stub_oracle": "9.9"}

    def known_value(self, knot, invariant):
        return self._value


def _wire(monkeypatch, stub):
    monkeypatch.setattr(registry, "_WIRED", (stub,))


def test_strict_passes_when_a_computed_oracle_agrees(monkeypatch):
    _wire(monkeypatch, _StubValidator(value=3))
    result = invariants.compute(knots.from_name("3_1"), "determinant")  # strict
    assert result.validation.verdict("stub_oracle") == "pass"
    assert result.validation.verdict("knotinfo") == "pass"
    assert result.validation.is_validated
    stub_record = next(
        v for v in result.validation.validators if v.oracle == "stub_oracle"
    )
    assert stub_record.version == "stub_oracle 9.9"


@pytest.mark.parametrize("mode", ["strict", "soft"])
def test_disagreeing_computed_oracle_raises_in_strict_and_soft(monkeypatch, mode):
    _wire(monkeypatch, _StubValidator(value=4))  # determinant of 3_1 is 3
    with pytest.raises(UnvalidatedResult, match="disagrees with stub_oracle"):
        invariants.compute(knots.from_name("3_1"), "determinant", validate=mode)


def test_strict_names_the_provisioning_gap_when_oracle_not_installed(monkeypatch):
    _wire(monkeypatch, _StubValidator(available=False))
    with pytest.raises(UnvalidatedResult) as excinfo:
        invariants.compute(knots.from_name("3_1"), "determinant")
    message = str(excinfo.value)
    assert "stub_oracle not installed" in message
    assert "install_oracles.sh" in message


def test_strict_raises_when_installed_oracle_cannot_check_this_input(monkeypatch):
    _wire(monkeypatch, _StubValidator(value=None))  # consulted, cannot check
    with pytest.raises(
        UnvalidatedResult, match="stub_oracle could not cross-check this input"
    ):
        invariants.compute(knots.from_name("3_1"), "determinant")


def test_soft_distinguishes_not_installed_in_its_info_message(monkeypatch, caplog):
    _wire(monkeypatch, _StubValidator(available=False))
    with caplog.at_level(logging.INFO, logger="tetradrome.invariants.compute"):
        result = invariants.compute(knots.from_name("3_1"), "determinant", validate="soft")
    assert result.validation.is_validated
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stub_oracle not installed" in m and "strict would have raised" in m
        for m in messages
    )


def test_not_run_verdict_is_recorded_for_an_oracle_that_cannot_check(monkeypatch):
    _wire(monkeypatch, _StubValidator(value=None))
    result = invariants.compute(
        knots.from_name("3_1"), "determinant", validate="soft"
    )
    assert result.validation.verdict("stub_oracle") == "not_run"
    assert result.validation.verdict("knotinfo") == "pass"


# --- registry surface ---

def test_registry_reports_kfh_wired_for_the_floer_invariants():
    for invariant in ("knot_floer_homology", "ozsvath_szabo_tau", "three_genus"):
        wired = registry.wired_validators(invariant)
        assert [v.name for v in wired] == ["knot_floer_homology"]
        assert registry.unwired_oracles(invariant) == ()
        assert registry.computed_oracle_exists(invariant)


def test_registry_reports_regina_wired_for_the_classical_three():
    for invariant in REGINA_COVERED:
        assert "regina" in [v.name for v in registry.wired_validators(invariant)]
        assert registry.unwired_oracles(invariant) == ("sage",)


def test_registry_reports_knotjob_wired_for_the_homological_three():
    for invariant in KNOTJOB_COVERED:
        assert "knotjob" in [v.name for v in registry.wired_validators(invariant)]
        assert "knotjob" not in registry.unwired_oracles(invariant)
        assert registry.unwired_oracles(invariant) != ()  # javakh/khoho/khoca remain


def test_registry_reports_the_rest_unwired():
    for invariant in STRICT_UNWIRED:
        assert registry.wired_validators(invariant) == ()
        assert registry.unwired_oracles(invariant) != ()
        assert registry.computed_oracle_exists(invariant)


def test_registry_reports_nothing_for_an_unknown_invariant():
    assert registry.wired_validators("homfly_polynomial") == ()
    assert registry.unwired_oracles("homfly_polynomial") == ()
    assert not registry.computed_oracle_exists("homfly_polynomial")
