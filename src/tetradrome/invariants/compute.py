"""Compute a knot invariant and return a validated, provenanced result.

Validate-by-default (decisions/0004): with validate=True a result that has no
validation path, or that disagrees with the oracle, raises UnvalidatedResult rather
than being returned.

Currently supports the Seifert-form invariants `determinant`, `signature`, and
`alexander_polynomial`, computed natively from the Seifert matrix and checked against
KnotInfo when the knot is tabulated. The Seifert matrix is itself computed natively
(Collins' algorithm) from the knot's braid word: a braid word supplied via
`from_braid` is used directly (off-table knots included); for a tabulated knot given
by name the braid word is read from KnotInfo. Off-table results have no oracle, so
under validate=True they raise.
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
    "alexander_polynomial": seifert.alexander_polynomial,
}


def compute(knot: NormalizedDiagram, invariant: str, validate: bool = True) -> InvariantResult:
    if invariant not in _SEIFERT_INVARIANTS:
        raise ValueError(
            f"compute does not support {invariant!r}; supported: {sorted(_SEIFERT_INVARIANTS)}"
        )

    if knot.braid is not None:
        braid = list(knot.braid)
        inputs = "braid_word"
    elif knot.identity is not None:
        braid = knotinfo_backend.braid_word(knot.identity)
        inputs = "knotinfo:braid_notation"
    else:
        raise UnknownKnot(
            "Invariant computation needs a braid word (from_braid) or a KnotInfo identity."
        )

    matrix = seifert.seifert_matrix_from_braid(braid)
    value = _SEIFERT_INVARIANTS[invariant](matrix)

    oracle = (
        knotinfo_backend.known_answer(knot.identity, invariant)
        if knot.identity is not None
        else None
    )
    if invariant == "alexander_polynomial" and oracle is not None:
        oracle = seifert.canonical_alexander(oracle)
    if oracle is None:
        known = "not_available"
    else:
        known = "pass" if value == oracle else "fail"

    result = InvariantResult(
        knot=knot.identity or "(braid word)",
        invariant=invariant,
        value=value,
        provenance=Provenance(
            backend="tetradrome-native",
            backend_version=__version__,
            method="seifert_form_from_braid",
            inputs=inputs,
        ),
        validation=ValidationStatus(known_answer_match=known),
    )

    if validate and known == "fail":
        raise UnvalidatedResult(
            f"{invariant} for {result.knot}: computed {value} disagrees with "
            f"KnotInfo oracle {oracle}."
        )
    if validate and not result.validation.is_validated:
        raise UnvalidatedResult(
            f"{invariant} for {result.knot}: no validation available "
            f"(known_answer_match={known})."
        )
    return result
