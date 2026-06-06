"""The result schema (SPEC 11).

Nothing is returned as a bare value: every invariant result carries how it was
produced and whether it was validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Provenance:
    """How a result was produced."""

    backend: str  # e.g. "tetradrome-native"
    backend_version: str
    method: str  # e.g. "seifert_form"
    inputs: str  # e.g. "knotinfo:seifert_matrix"


@dataclass(frozen=True)
class ValidationStatus:
    """The outcome of each validation check (SPEC 11)."""

    known_answer_match: str = "not_available"  # pass | fail | not_available
    independent_backend_match: str = "not_run"  # pass | fail | not_run
    d_squared_check: str = "not_applicable"  # pass | fail | not_applicable

    @property
    def is_validated(self) -> bool:
        return "pass" in (self.known_answer_match, self.independent_backend_match)


@dataclass(frozen=True)
class InvariantResult:
    """A computed invariant with full provenance and validation status."""

    knot: str | None
    invariant: str
    value: Any
    provenance: Provenance
    validation: ValidationStatus
    raw: Any = None
