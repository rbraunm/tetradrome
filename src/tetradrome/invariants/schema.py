# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The result schema (SPEC 11).

Nothing is returned as a bare value: every invariant result carries how it was
produced and whether it was validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Provenance:
    """How a result was produced (ADR 0013).

    Records the backend and its version, the method, the inputs, and the versions of the
    computational libraries that can move the answer. For a pure-native computation that library
    set is just the interpreter -- which is the point: nothing external moved the number (ADR 0006),
    so the provenance chain is as short as it can honestly be.
    """

    backend: str  # e.g. "tetradrome-native"
    backend_version: str
    method: str  # e.g. "seifert_form"
    inputs: str  # e.g. "knotinfo:seifert_matrix"
    library_versions: tuple[tuple[str, str], ...] = ()  # (name, version), e.g. (("python", "3.12.3"),)


@dataclass(frozen=True)
class ValidatorRecord:
    """One validator's verdict on a result (ADR 0013).

    KnotInfo is recorded as one of these, never as a special case.
    """

    oracle: str  # "knotinfo", "regina", "knot_floer_homology", ...
    version: str  # the exact version consulted
    verdict: str  # pass | fail | not_run


@dataclass(frozen=True)
class ValidationStatus:
    """The validators consulted for a result and their verdicts (SPEC 11, ADR 0013).

    A result is validated when at least one validator passed; it has a disagreement when any
    validator that ran disagreed (the fail-loud trigger). ``d_squared_check`` is deliberately
    separate: it is the native complex's own d^2 = 0 self-consistency check, not an external
    validator.
    """

    validators: tuple[ValidatorRecord, ...] = ()
    d_squared_check: str = "not_applicable"  # pass | fail | not_applicable

    @property
    def is_validated(self) -> bool:
        return any(v.verdict == "pass" for v in self.validators)

    @property
    def has_disagreement(self) -> bool:
        return any(v.verdict == "fail" for v in self.validators)

    def verdict(self, oracle: str) -> str:
        """This oracle's verdict (pass | fail | not_run); not_run if it was not consulted."""
        return next((v.verdict for v in self.validators if v.oracle == oracle), "not_run")


@dataclass(frozen=True)
class InvariantResult:
    """A computed invariant with full provenance and validation status."""

    knot: str | None
    invariant: str
    value: Any
    provenance: Provenance
    validation: ValidationStatus
    raw: Any = None
