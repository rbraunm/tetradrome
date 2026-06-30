# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Measurement adapters for the comparison artifact.

Two kinds, behind one rule: only what is actually present is timed; an absent tool reports its
absence, never a guessed number (CLAUDE.md: the artifact reports data, never fakes it).

  * Tetradrome side -- ``measureTetradrome`` times ``invariants.compute`` for the invariants that
    are implemented, and carries the validation status the result already knows (so the artifact
    doubles as a known-answer check).
  * Oracle side -- one probe + one timed call per external tool. ``knot_floer_homology`` is wired
    for real (it is pip-installable and standalone). SnapPy / KnotJob / Sage / Khoca are probed for
    presence only; their timed calls are filled in once a host (CT 250) has them installed and the
    real invocation is confirmed -- until then they report "adapter pending".

Timing is best-of-``reps`` wall seconds (the floor is the least noisy estimate of compute cost).
"""
from __future__ import annotations

import dataclasses
import importlib
import shutil
import time


@dataclasses.dataclass
class Measurement:
    value: str                  # short repr of the computed value (for the cross-check column)
    seconds: float | None       # best-of-reps wall seconds; None if not measured
    note: str = ""              # e.g. "same pd_to_hfk call", "adapter pending", "absent"
    agree: str = ""             # pass | mirror | mismatch | oracle | n/a | ""


def _best(callable_, reps):
    """Best-of-``reps`` wall seconds for ``callable_()``; returns (result, seconds)."""
    floor = None
    result = None
    for _ in range(max(reps, 1)):
        start = time.perf_counter()
        result = callable_()
        elapsed = time.perf_counter() - start
        if floor is None or elapsed < floor:
            floor = elapsed
    return result, floor


# ---- knot construction & PD ---------------------------------------------------------------

def buildLadder(names):
    """[(name, knot)] for tabulated knots; a knot exposes ``pd_code`` and ``identity``."""
    from tetradrome import knots
    ladder = []
    for name in names:
        ladder.append((name, knots.from_name(name)))
    return ladder


def pdAsList(knot):
    """kfh wants a list of tuples (or a PD string), not Tetradrome's tuple-of-tuples."""
    return [list(crossing) for crossing in knot.pd_code]


# ---- Tetradrome side ----------------------------------------------------------------------

def measureTetradrome(knot, computeName, reps):
    """Time ``invariants.compute(knot, computeName)`` and capture its validation verdict."""
    from tetradrome import invariants
    try:
        result, seconds = _best(lambda: invariants.compute(knot, computeName), reps)
    except Exception as error:                       # an engine that cannot run on this knot
        return Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:60], agree="n/a")
    match = getattr(result.validation, "known_answer_match", "")
    agree = {"pass": "pass", "not_available": "no-oracle"}.get(match, match or "")
    return Measurement(value=_shortValue(result.value), seconds=seconds, agree=agree)


def _shortValue(value):
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


# ---- native grid (knot Floer) engine ------------------------------------------------------

_MACHINE = None


def _machine():
    global _MACHINE
    if _MACHINE is None:
        from tetradrome.scheduler import detect_machine
        _MACHINE = detect_machine()
    return _MACHINE


def measureFloerGrid(knotName, reps):
    """Time Tetradrome's native grid (knot Floer) engine end to end for a tabulated knot: build
    the grid, build the Poincare graph, run it through the scheduler. The engine is expensive and
    deterministic, so a single timed run is taken (``reps`` is intentionally not applied here).
    Fails loud on an infeasible or failed run rather than reporting a partial time. The scheduler
    spawns worker processes, so this must run from a real module with a __main__ guard (the
    generator is); it will hang if driven from a REPL or a stdin heredoc."""
    from tetradrome.engines.floer import GridDiagram
    from tetradrome.engines.floer.scheduling import whole_knot_graph
    from tetradrome.scheduler import Scheduler
    machine = _machine()
    grid = GridDiagram.from_knotinfo(knotName)
    start = time.perf_counter()
    graph, key = whole_knot_graph(grid, backend="bitint")
    report = Scheduler(machine).run(graph)
    seconds = time.perf_counter() - start
    if report.infeasible:
        return Measurement(value="infeasible", seconds=None,
                           note=str(report.infeasible[0])[:60], agree="n/a")
    if report.failures:
        return Measurement(value="failed", seconds=None,
                           note=str(report.failures[0])[:60], agree="n/a")
    # The support count is recorded for a future HFK-rank cross-check (confirmed on the box, not
    # here); the artifact does not yet claim a live rank match for Floer, so agree stays neutral.
    return Measurement(value=f"support={len(report.results[key])}", seconds=seconds, agree="n/a")


# ---- knot_floer_homology (real) -----------------------------------------------------------

def kfhAvailable():
    try:
        importlib.import_module("knot_floer_homology")
        return True, "knot_floer_homology"
    except Exception:
        return False, "knot_floer_homology not importable"


# What a single pd_to_hfk call yields -> our invariant names.
KFH_FIELDS = {
    "tau": "tau",
    "seifert_genus": "seifert_genus",
    "fibered": "fibered",
    "epsilon": "epsilon",
    "nu": "nu",
    "l_space": "l_space_knot",
}


def kfhRun(knot, reps):
    """One timed pd_to_hfk call; returns {invariantName: Measurement}. The HFK ranks carry the
    measured time; the scalar invariants come from the SAME call and are noted as such."""
    kfh = importlib.import_module("knot_floer_homology")
    pd = pdAsList(knot)
    try:
        out, seconds = _best(lambda: kfh.pd_to_hfk(pd), reps)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:60], agree="n/a")
        return {"hfk": miss}
    results = {}
    total = out.get("total_rank")
    results["hfk"] = Measurement(value=f"total_rank={total}", seconds=seconds, agree="oracle")
    for invName, field in KFH_FIELDS.items():
        results[invName] = Measurement(value=f"{out.get(field)}", seconds=None,
                                       note="same pd_to_hfk call", agree="oracle")
    return results


# ---- probe-only oracles (timed calls land once a host has them) ---------------------------

def _probeImport(moduleName, label):
    try:
        importlib.import_module(moduleName)
        return True, label
    except Exception:
        return False, f"{label} not importable"


def _probeBinary(binaryName, label):
    path = shutil.which(binaryName)
    return (bool(path), f"{label} at {path}" if path else f"{label} not on PATH")


def snappyAvailable():
    return _probeImport("snappy", "SnapPy")


def knotjobAvailable():
    return _probeBinary("knotjob", "KnotJob")


def sageAvailable():
    return _probeBinary("sage", "SageMath")


def khocaAvailable():
    # Khoca ships as a CLI and/or a python module; probe both.
    ok, detail = _probeBinary("khoca", "Khoca")
    if ok:
        return ok, detail
    return _probeImport("khoca", "Khoca")


# Registry the generator iterates. ``run`` is None for probe-only oracles -- the generator prints
# "adapter pending" for the invariants they cover until a run is wired against a real install.
@dataclasses.dataclass
class Oracle:
    key: str
    available: object           # () -> (bool, detail)
    run: object | None          # (knot, reps) -> {invariantName: Measurement} | None


ORACLES = [
    Oracle("kfh", kfhAvailable, kfhRun),
    Oracle("snappy", snappyAvailable, None),
    Oracle("knotjob", knotjobAvailable, None),
    Oracle("sage", sageAvailable, None),
    Oracle("khoca", khocaAvailable, None),
]
