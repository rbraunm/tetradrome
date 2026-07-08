# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Compute a knot invariant and return a validated, provenanced result.

Validate-by-default (decisions/0004): with validate=True a result that has no
validation path, or that disagrees with the oracle, raises UnvalidatedResult rather
than being returned.

Supported invariants: the Seifert-form invariants `determinant`, `signature`, and
`alexander_polynomial` (from the Collins braid Seifert matrix); `jones_polynomial` (from
the Kauffman bracket over the resolution cube); and the native homological invariants
`khovanov_homology` (over F2), `rational_khovanov_homology` (over Q), and `rasmussen_s`
(from the Lee quantum filtration). All are checked against KnotInfo when the knot is
tabulated -- the homological oracles are mirrored/sign-flipped to KnotInfo's chirality
convention (Phase 2c/3c). Seifert-form invariants accept a braid word (from_braid,
off-table included) or a tabulated knot's KnotInfo braid; the diagrammatic invariants
need a PD diagram (from_name or from_pd). Off-table results have no oracle, so under
validate=True they raise.
"""
from __future__ import annotations

from .._version import __version__
from ..backends import knotinfo_backend
from ..diagrams import NormalizedDiagram
from ..engines import khovanov
from ..errors import UnknownKnot, UnvalidatedResult
from . import jones, seifert
from .schema import InvariantResult, Provenance, ValidationStatus, ValidatorRecord

# Invariants read from the braid Seifert matrix (Collins).
_SEIFERT_INVARIANTS = {
    "determinant": seifert.determinant,
    "signature": seifert.signature,
    "alexander_polynomial": seifert.alexander_polynomial,
}
# Invariants read from the PD diagram via the resolution cube: (function, method label).
_PD_INVARIANTS = {
    "jones_polynomial": (jones.jones_polynomial, "kauffman_bracket"),
    "khovanov_homology": (khovanov.khovanov_homology, "khovanov_cube_f2"),
    "rational_khovanov_homology": (khovanov.khovanov_homology_q, "khovanov_cube_q"),
    "rasmussen_s": (khovanov.rasmussen_s, "lee_quantum_filtration"),
}
# PD invariants whose computation verifies d^2 = 0 over its coefficient ring.
_RUNS_D_SQUARED = {"khovanov_homology", "rational_khovanov_homology", "rasmussen_s"}


def _library_versions(used: tuple[str, ...] = ()) -> tuple[tuple[str, str], ...]:
    """Versions of the computational libraries a result depends on (ADR 0013).

    Always records the interpreter, then each library the computation actually used. The native
    core uses no third-party computational library (ADR 0006), so a native result records only the
    interpreter -- that short chain is the native-first argument made concrete, and it is the true
    record, not the set of libraries that merely happen to be installed. A declared library that is
    missing is a fail-loud error (the caller claimed to use something absent), never a silent skip.
    """
    import platform
    from importlib.metadata import version as _packageVersion

    versions: list[tuple[str, str]] = [("python", platform.python_version())]
    for name in used:
        versions.append((name, _packageVersion(name)))
    return tuple(versions)


def _finalize(knot, invariant, value, method, inputs, fallback_label, validate,
              d_squared="not_applicable", libraries: tuple[str, ...] = ()):
    """Attach the oracle check, provenance, and validation outcome to a value.

    ``libraries`` names the third-party computational libraries the computation used, if any; the
    native core uses none, so it defaults to empty and the result records only the interpreter.
    """
    oracle = (
        knotinfo_backend.known_answer(knot.identity, invariant)
        if knot.identity is not None
        else None
    )
    if oracle is not None:
        if invariant == "alexander_polynomial":
            oracle = seifert.canonical_alexander(oracle)
        elif invariant == "jones_polynomial":
            oracle = jones.canonical_laurent(*oracle)

    # KnotInfo is the only validator wired at this checkpoint; it is recorded as one validator
    # among what will become many (ADR 0013). No tabulated answer means KnotInfo was not consulted.
    if oracle is None:
        validators: tuple[ValidatorRecord, ...] = ()
    else:
        verdict = "pass" if value == oracle else "fail"
        validators = (ValidatorRecord("knotinfo", knotinfo_backend.version(), verdict),)

    result = InvariantResult(
        knot=knot.identity or fallback_label,
        invariant=invariant,
        value=value,
        provenance=Provenance(
            backend="tetradrome-native",
            backend_version=__version__,
            method=method,
            inputs=inputs,
            library_versions=_library_versions(libraries),
        ),
        validation=ValidationStatus(validators=validators, d_squared_check=d_squared),
    )

    if validate and result.validation.has_disagreement:
        disagreeing = next(v for v in validators if v.verdict == "fail")
        raise UnvalidatedResult(
            f"{invariant} for {result.knot}: computed {value!r} disagrees with "
            f"{disagreeing.oracle} {disagreeing.version} oracle {oracle!r}."
        )
    if validate and not result.validation.is_validated:
        raise UnvalidatedResult(
            f"{invariant} for {result.knot}: no validator passed "
            f"(validators={validators})."
        )
    return result


def compute(knot: NormalizedDiagram, invariant: str, validate: bool = True) -> InvariantResult:
    if invariant in _SEIFERT_INVARIANTS:
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
        value = _SEIFERT_INVARIANTS[invariant](seifert.seifert_matrix_from_braid(braid))
        return _finalize(
            knot, invariant, value, "seifert_form_from_braid", inputs, "(braid word)", validate
        )

    if invariant in _PD_INVARIANTS:
        if not knot.pd_code:
            raise UnknownKnot(
                f"{invariant} needs a PD diagram (from_name or from_pd); "
                "this knot is braid-presented."
            )
        func, method = _PD_INVARIANTS[invariant]
        value = func(knot.pd_code)
        d_squared = "pass" if invariant in _RUNS_D_SQUARED else "not_applicable"
        return _finalize(
            knot, invariant, value, method, "pd_code", "(pd)", validate, d_squared=d_squared
        )

    supported = sorted(_SEIFERT_INVARIANTS) + sorted(_PD_INVARIANTS)
    raise ValueError(f"compute does not support {invariant!r}; supported: {supported}")
