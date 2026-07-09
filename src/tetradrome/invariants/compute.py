# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Compute a knot invariant and return a validated, provenanced result.

Validate-by-default (decisions/0004): ``validate`` is a three-mode enum, default
"strict". Strict requires a computed oracle wherever one exists anywhere (the registry
knows), and the raise distinguishes "not installed" from "not yet wired"; soft
tolerates a computed oracle's absence (falling back to KnotInfo with an info message)
but never a disagreement; off skips validation entirely while keeping full provenance.
In every validating mode, ANY oracle that runs and disagrees raises UnvalidatedResult.

Supported invariants: the Seifert-form invariants `determinant`, `signature`, and
`alexander_polynomial` (from the Collins braid Seifert matrix); `jones_polynomial` (from
the Kauffman bracket over the resolution cube); and the native homological invariants
`khovanov_homology` (over F2), `rational_khovanov_homology` (over Q), and `rasmussen_s`
(from the Lee quantum filtration); and the native grid Floer invariants
`knot_floer_homology` (HFK-hat), `ozsvath_szabo_tau`, and `three_genus`. All are checked
against KnotInfo when the knot is tabulated -- the Khovanov/s oracles are
mirrored/sign-flipped to KnotInfo's chirality convention (Phase 2c/3c), while the Floer
oracles compare RAW: KnotInfo's HFK/tau/genus columns share its PD chirality, and the
native grid is built in that standard chirality, so any mismatch is a genuine error and
raises -- never a silent reflect/transpose. Seifert-form invariants accept a braid word
(from_braid, off-table included) or a tabulated knot's KnotInfo braid; the diagrammatic
invariants need a PD diagram (from_name or from_pd); the Floer invariants need a
tabulated knot (the grid comes from KnotInfo's grid notation). Off-table results have no KnotInfo oracle,
so until their computed oracles are wired they raise under strict and soft; pass
validate="off" to opt in.
"""
from __future__ import annotations

import logging
from typing import Literal

from .._version import __version__
from ..backends import knotinfo_backend, registry
from ..diagrams import NormalizedDiagram
from ..engines import floer, khovanov
from ..errors import UnknownKnot, UnvalidatedResult
from . import jones, seifert
from .schema import InvariantResult, Provenance, ValidationStatus, ValidatorRecord

logger = logging.getLogger(__name__)

_VALIDATE_MODES = ("strict", "soft", "off")

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
# Invariants read from the grid diagram via the native Floer engine: (function, method label).
# The grid comes from KnotInfo's tabulated grid notation, so these are tabulated-knot-only
# paths. The engine carries its own fail-loud self-checks (hfk_hat verifies the V-factor
# division by reconstruction; tau verifies dim H_0 = 1), which are not d^2 checks, so
# d_squared_check stays not_applicable here.
_FLOER_INVARIANTS = {
    "knot_floer_homology": (floer.hfk_hat, "grid_hfk_hat"),
    "ozsvath_szabo_tau": (floer.tau, "grid_filtered_tau"),
    "three_genus": (floer.seifert_genus, "grid_hfk_genus"),
}


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


def _computed_oracle_gap(invariant, not_installed, could_not_check):
    """Why no computed oracle checked this result: provisioning gap vs dev gap (ADR 0004)."""
    parts = []
    if not_installed:
        parts.append(
            f"{', '.join(not_installed)} not installed (run scripts/install_oracles.sh)"
        )
    if could_not_check:
        parts.append(f"{', '.join(could_not_check)} could not cross-check this input")
    unwired = registry.unwired_oracles(invariant)
    if unwired:
        parts.append(f"{', '.join(unwired)} not yet wired")
    return "; ".join(parts)


def _finalize(knot, invariant, value, method, inputs, fallback_label, validate,
              d_squared="not_applicable", libraries: tuple[str, ...] = ()):
    """Attach the validator verdicts, provenance, and validation outcome to a value.

    Every wired + installed validator covering the invariant is consulted (registry), then
    KnotInfo rides along as one more record -- an extra cross-check, never the primary
    validator (ADR 0004). ``libraries`` names the third-party computational libraries the
    computation used, if any; the native core uses none, so it defaults to empty and the
    result records only the interpreter.
    """
    records: list[ValidatorRecord] = []
    known_values: dict[str, object] = {}
    not_installed: list[str] = []
    could_not_check: list[str] = []

    if validate != "off":
        for validator in registry.wired_validators(invariant):
            if not validator.is_available():
                not_installed.append(validator.name)
                continue
            version = registry.flat_version(validator)
            known = validator.known_value(knot, invariant)
            if known is None:
                # Consulted, but it cannot check this input -- honest record, never a guess.
                records.append(ValidatorRecord(validator.name, version, "not_run"))
                could_not_check.append(validator.name)
                continue
            known_values[validator.name] = known
            records.append(
                ValidatorRecord(validator.name, version, "pass" if value == known else "fail")
            )

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
            known_values["knotinfo"] = oracle
            records.append(
                ValidatorRecord(
                    "knotinfo", knotinfo_backend.version(), "pass" if value == oracle else "fail"
                )
            )

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
        validation=ValidationStatus(validators=tuple(records), d_squared_check=d_squared),
    )

    if validate == "off":
        return result

    # Strict and soft alike: ANY oracle that ran and disagrees raises. Soft tolerates
    # absence, never disagreement (ADR 0004).
    if result.validation.has_disagreement:
        disagreeing = next(v for v in records if v.verdict == "fail")
        raise UnvalidatedResult(
            f"{invariant} for {result.knot}: computed {value!r} disagrees with "
            f"{disagreeing.oracle} {disagreeing.version} oracle "
            f"{known_values[disagreeing.oracle]!r}."
        )

    if any(v.verdict == "pass" and v.oracle != "knotinfo" for v in records):
        return result  # a computed oracle passed: validated in every mode

    knotinfo_passed = result.validation.verdict("knotinfo") == "pass"
    if not registry.computed_oracle_exists(invariant):
        # No computed oracle exists anywhere: KnotInfo is the fallback of last resort.
        if knotinfo_passed:
            return result
        raise UnvalidatedResult(
            f"{invariant} for {result.knot}: no validator passed "
            f"(validators={tuple(records)})."
        )

    gap = _computed_oracle_gap(invariant, not_installed, could_not_check)
    if validate == "strict":
        raise UnvalidatedResult(
            f"{invariant} for {result.knot}: strict validation requires a computed "
            f"oracle; {gap}."
        )
    # soft: the computed oracle's absence is tolerated; KnotInfo must carry the result.
    if knotinfo_passed:
        logger.info(
            "%s for %s: soft validation passed on KnotInfo only; strict would have "
            "raised (%s).",
            invariant, result.knot, gap,
        )
        return result
    raise UnvalidatedResult(
        f"{invariant} for {result.knot}: no computed oracle ran ({gap}) and KnotInfo "
        f"has no value for it."
    )


def compute(
    knot: NormalizedDiagram,
    invariant: str,
    validate: Literal["strict", "soft", "off"] = "strict",
) -> InvariantResult:
    if validate not in _VALIDATE_MODES:
        raise ValueError(
            f"validate must be one of {_VALIDATE_MODES}, got {validate!r}; the boolean "
            f"form is gone (True -> 'strict', False -> 'off')."
        )
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

    if invariant in _FLOER_INVARIANTS:
        if knot.identity is None:
            raise UnknownKnot(
                f"{invariant} needs a tabulated knot: the grid diagram comes from "
                "KnotInfo's grid notation, and an off-table knot has no grid."
            )
        grid = floer.GridDiagram.from_knotinfo(knot.identity)
        func, method = _FLOER_INVARIANTS[invariant]
        value = func(grid)
        return _finalize(
            knot, invariant, value, method, "knotinfo:grid_notation", "(grid)", validate
        )

    supported = sorted(_SEIFERT_INVARIANTS) + sorted(_PD_INVARIANTS) + sorted(_FLOER_INVARIANTS)
    raise ValueError(f"compute does not support {invariant!r}; supported: {supported}")
