"""Compute a knot invariant and return a validated, provenanced result.

Validate-by-default (decisions/0004): with validate=True a result that has no
validation path, or that disagrees with the oracle, raises UnvalidatedResult rather
than being returned.

Currently supports the Seifert-form invariants `determinant` and `signature`,
computed natively and checked against KnotInfo. The Seifert matrix is sourced from
KnotInfo for tabulated knots; a native Seifert algorithm (for off-table knots) is
future work.
"""
from __future__ import annotations

from .._version import __version__
from ..backends import knotinfo_backend
from ..diagrams import NormalizedDiagram
from ..errors import UnknownKnot, UnvalidatedResult
from . import seifert
from .schema import InvariantResult, Provenance, ValidationStatus

_SEIFERT_INVARIANTS = {
    "determinant": seifert.determinant,
    "signature": seifert.signature,
}


def compute(knot: NormalizedDiagram, invariant: str, validate: bool = True) -> InvariantResult:
    if invariant not in _SEIFERT_INVARIANTS:
        raise ValueError(
            f"compute does not support {invariant!r}; supported: {sorted(_SEIFERT_INVARIANTS)}"
        )
    if knot.identity is None:
        raise UnknownKnot(
            "Invariant computation currently needs a KnotInfo-identified knot "
            "(no native Seifert algorithm for off-table knots yet)."
        )

    matrix = knotinfo_backend.seifert_matrix(knot.identity)
    value = _SEIFERT_INVARIANTS[invariant](matrix)

    oracle = knotinfo_backend.known_answer(knot.identity, invariant)
    if oracle is None:
        known = "not_available"
    else:
        known = "pass" if value == oracle else "fail"

    result = InvariantResult(
        knot=knot.identity,
        invariant=invariant,
        value=value,
        provenance=Provenance(
            backend="tetradrome-native",
            backend_version=__version__,
            method="seifert_form",
            inputs="knotinfo:seifert_matrix",
        ),
        validation=ValidationStatus(known_answer_match=known),
    )

    if validate and known == "fail":
        raise UnvalidatedResult(
            f"{invariant} for {knot.identity}: computed {value} disagrees with "
            f"KnotInfo oracle {oracle}."
        )
    if validate and not result.validation.is_validated:
        raise UnvalidatedResult(
            f"{invariant} for {knot.identity}: no validation available "
            f"(known_answer_match={known})."
        )
    return result
