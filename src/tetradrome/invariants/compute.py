# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Compute a knot invariant and return a validated, provenanced result.

Validate-by-default: ``validate`` is a three-mode enum, default
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
invariants need a PD diagram (from_name or from_pd); the Floer invariants take any knot, its
grid built from the PD diagram or the braid's closure. Off-table results have no KnotInfo oracle,
so until their computed oracles are wired they raise under strict and soft; pass
validate="off" to opt in.
"""
from __future__ import annotations

import logging
from typing import Literal

from .._version import __version__
from ..backends import knotinfo_backend, registry
from ..diagrams import NormalizedDiagram, build
from ..diagrams.braid_closure import pd_from_braid
from ..engines import floer, khovanov
from ..engines.floer import grid_store
from ..errors import UnknownKnot, UnvalidatedResult
from ..scheduler import ComputePath, Job, JobGraph, Placement, recording, run_to_result
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
# The grid is built from the knot's crossing diagram, or its braid's closure, and kept in the
# grid store (engines/floer/grid_store.py). The engine carries its own fail-loud self-checks
# (hfk_hat verifies the V-factor division by reconstruction; tau verifies dim H_0 = 1), which
# are not d^2 checks, so d_squared_check stays not_applicable here.
_FLOER_INVARIANTS = {
    "knot_floer_homology": (floer.hfk_hat, "grid_hfk_hat"),
    "ozsvath_szabo_tau": (floer.tau, "grid_filtered_tau"),
    "three_genus": (floer.seifert_genus, "grid_hfk_genus"),
}


# What a worker dirties before it holds any of the calculation: the scheduler's measured figure
# for a pre-warmed worker, as engines/floer/scheduling.py prices its slices.
_WORKER_BASE_BYTES = 16 * 2**20
# A cube calculation holds on the order of its 2^c resolution states; priced per state, and
# tuned, like every footprint, by the scheduler's over-budget warnings. The two that reduce over
# the rationals hold exact fractions: measured at 64 MB for the 1,024 states of a ten-crossing
# knot and 34 MB for 256 states.
_CUBE_STATE_BYTES = {
    "jones_polynomial": 2048,
    "khovanov_homology": 2048,
    "rational_khovanov_homology": 56 * 1024,
    "rasmussen_s": 56 * 1024,
}
_SEIFERT_ENTRY_BYTES = 256


def _seifert_run(inputs, deps):
    invariant, braid = inputs
    return _SEIFERT_INVARIANTS[invariant](seifert.seifert_matrix_from_braid(braid))


def _pd_run(inputs, deps):
    invariant, pd_code = inputs
    return _PD_INVARIANTS[invariant][0](pd_code)


def _job(invariant: str, run, inputs, cost: float, working_bytes: int) -> Job:
    """One calculation as a scheduler job. It can run in this process or in a worker; the
    scheduler chooses by the benchmark's figure for this invariant (its own kind)."""
    return Job(
        key=(invariant,), run=run, inputs=inputs, kind=f"invariant.{invariant}", cost=cost,
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=working_bytes),
               ComputePath(Placement.CPU_PINNED, cores=1,
                           ram_bytes=_WORKER_BASE_BYTES + working_bytes)),
    )


def _seifert_job(invariant: str, braid) -> Job:
    strands = max((abs(generator) for generator in braid), default=0) + 1
    size = len(braid) + strands                      # the Seifert matrix is at most this wide
    return _job(invariant, _seifert_run, (invariant, tuple(braid)),
                cost=float(size ** 3), working_bytes=size * size * _SEIFERT_ENTRY_BYTES)


def _pd_job(invariant: str, pd_code) -> Job:
    states = 2 ** len(pd_code)                       # the resolution cube's vertices
    return _job(invariant, _pd_run, (invariant, pd_code),
                cost=float(states), working_bytes=states * _CUBE_STATE_BYTES[invariant])


def _scheduled(job: Job):
    return run_to_result(JobGraph([job]), job.key)


# The knot the benchmark computes the Seifert and cube invariants of: six crossings, between the
# smallest table knots and the ones whose cube takes most of a second.
_BENCHMARK_KNOT = "6_2"


def benchmark(run) -> None:
    """The Seifert and cube engines' representative calculations for ``tetradrome.calibrate``:
    every invariant of one knot, each run through ``run(graph, key)``."""
    knot = build.from_name(_BENCHMARK_KNOT)
    braid = knotinfo_backend.braid_word(knot.identity)
    jobs = [_seifert_job(invariant, braid) for invariant in _SEIFERT_INVARIANTS]
    jobs += [_pd_job(invariant, knot.pd_code) for invariant in _PD_INVARIANTS]
    for job in jobs:
        run(JobGraph([job]), job.key)


def _supported() -> list[str]:
    """Every invariant name compute() accepts -- the single source for that set, read by
    the unsupported-name error below and by the tests pinning the canonical-name record."""
    return sorted(_SEIFERT_INVARIANTS) + sorted(_PD_INVARIANTS) + sorted(_FLOER_INVARIANTS)


def _library_versions(used: tuple[str, ...] = ()) -> tuple[tuple[str, str], ...]:
    """Versions of the computational libraries a result depends on.

    Always records the interpreter, then each library the computation actually used: the true
    record, not the set of libraries that merely happen to be installed. The Seifert-form and cube
    engines use none, so their results record only the interpreter; the grid engine's kernels run
    on numpy and, when it is installed, Numba. A declared library that is missing is a fail-loud
    error (the caller claimed to use something absent), never a silent skip.
    """
    import platform
    from importlib.metadata import version as _packageVersion

    versions: list[tuple[str, str]] = [("python", platform.python_version())]
    for name in used:
        versions.append((name, _packageVersion(name)))
    return tuple(versions)


def _floer_libraries() -> tuple[str, ...]:
    """The numerical libraries the grid engine's block kernels run on."""
    from ..engines.floer.differential_jit import HAVE_NUMBA

    return ("numpy", "numba") if HAVE_NUMBA else ("numpy",)


def _computed_oracle_gap(invariant, not_installed, could_not_check):
    """Why no computed oracle checked this result: provisioning gap vs dev gap."""
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


def _finalize(knot, invariant, value, method, inputs, fallback_label, validate, computation,
              d_squared="not_applicable", libraries: tuple[str, ...] = (), grid=None):
    """Attach the validator verdicts, provenance, and validation outcome to a value.

    Every wired + installed validator covering the invariant is consulted (registry), then
    KnotInfo rides along as one more record -- an extra cross-check, never the primary
    validator. ``libraries`` names the third-party computational libraries the
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
            grid=grid,
            computation=computation,
        ),
        validation=ValidationStatus(validators=tuple(records), d_squared_check=d_squared),
    )

    if validate == "off":
        return result

    # Strict and soft alike: ANY oracle that ran and disagrees raises. Soft tolerates
    # absence, never disagreement.
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
    with recording() as recorder:
        calculated = _calculated(knot, invariant)
    return _finalize(knot, invariant, validate=validate, computation=recorder.record(),
                     **calculated)


def _calculated(knot: NormalizedDiagram, invariant: str) -> dict:
    """The value and how it was reached, as ``_finalize``'s keyword arguments."""
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
        return {"value": _scheduled(_seifert_job(invariant, braid)),
                "method": "seifert_form_from_braid", "inputs": inputs,
                "fallback_label": "(braid word)"}

    if invariant in _PD_INVARIANTS:
        if not knot.pd_code:
            raise UnknownKnot(
                f"{invariant} needs a PD diagram (from_name or from_pd); "
                "this knot is braid-presented."
            )
        _, method = _PD_INVARIANTS[invariant]
        return {"value": _scheduled(_pd_job(invariant, knot.pd_code)), "method": method,
                "inputs": "pd_code", "fallback_label": "(pd)",
                "d_squared": "pass" if invariant in _RUNS_D_SQUARED else "not_applicable"}

    if invariant in _FLOER_INVARIANTS:
        if knot.pd_code:
            pd, inputs, fallback_label = knot.pd_code, "pd_code", "(pd)"
        else:
            pd, inputs, fallback_label = pd_from_braid(knot.braid), "braid_word", "(braid word)"
        located = grid_store.knot_grid(pd)
        if knot.identity is not None:
            smallest_known = len(knotinfo_backend.grid_notation(knot.identity)) // 2
            if located.size > smallest_known:
                logger.warning(
                    "%s: the grid built for %s has %d rows; KnotInfo records one of %d, so this "
                    "calculation does more work than the knot needs.",
                    invariant, knot.identity, located.size, smallest_known)
        func, method = _FLOER_INVARIANTS[invariant]
        return {"value": func(located.grid), "method": method, "inputs": inputs,
                "fallback_label": fallback_label, "libraries": _floer_libraries(),
                "grid": located}

    raise ValueError(f"compute does not support {invariant!r}; supported: {_supported()}")
