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
import re
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


# ---- shared oracle normalization: Khovanov polynomials, mirror, agreement ------------------
#
# Several oracles (KnotJob, JavaKh, KhoHo, ...) emit Khovanov homology as a Laurent polynomial in
# t (homological) and q (quantum). We parse those to {(h, q): rank} -- the shape
# ``invariants.compute`` returns natively -- then judge agreement up to the per-oracle mirror the
# design pass fixed: a knot read with the opposite PD/orientation convention is the mirror knot,
# whose Khovanov is (h, q) -> (-h, -q) and whose Rasmussen s negates. So a verdict is "pass"
# (equal outright, e.g. an amphichiral knot), "mirror" (equal after the transform -- the expected
# result for a mirror-convention oracle), or "mismatch" (a real disagreement).

def _bracketPD(knot):
    """Tetradrome's PD as the ``PD[X[...],...]`` bracket string KnotJob / JavaKh read."""
    return "PD[" + ",".join("X[%d,%d,%d,%d]" % tuple(crossing) for crossing in knot.pd_code) + "]"


def _monomial(term):
    """One ``c t^h q^q`` monomial (factors ``*``-joined or juxtaposed, exponents optional and
    possibly negative) -> ((h, q), coefficient)."""
    term = term.replace("*", "").strip()
    leading = re.match(r"[+-]?\d+", term)
    coefficient = int(leading.group(0)) if leading else 1
    tPart = re.search(r"t(\^-?\d+)?", term)
    h = 0 if tPart is None else (int(tPart.group(1)[1:]) if tPart.group(1) else 1)
    qPart = re.search(r"q(\^-?\d+)?", term)
    q = 0 if qPart is None else (int(qPart.group(1)[1:]) if qPart.group(1) else 1)
    return (h, q), coefficient


def _parseKhovanovPoly(text):
    """Sum of Khovanov monomials in t, q -> {(h, q): rank} (zeros dropped). Handles ``*`` or
    juxtaposed factors, negative exponents, and KhoHo's parenthesized (h=0) groups."""
    text = text.replace("(", "").replace(")", "").replace(" ", "")
    groups: dict = {}
    for term in text.split("+"):
        if not term:
            continue
        key, coefficient = _monomial(term)
        groups[key] = groups.get(key, 0) + coefficient
    return {key: c for key, c in groups.items() if c}


def _mirrorKhovanov(groups):
    """Khovanov of the mirror knot: (h, q) -> (-h, -q)."""
    return {(-h, -q): rank for (h, q), rank in groups.items()}


def _f2FromIntegral(free, torsion):
    """Khovanov dimensions over F2 from the integral free ranks plus the order-2 torsion, by the
    universal coefficient theorem: a Z/2 summand at (h, q) gives an F2 class at (h, q) and at
    (h-1, q). Tetradrome's ``khovanov_homology`` is this F2 theory."""
    f2 = dict(free)
    for (h, q), count in torsion.items():
        f2[(h, q)] = f2.get((h, q), 0) + count
        f2[(h - 1, q)] = f2.get((h - 1, q), 0) + count
    return {key: rank for key, rank in f2.items() if rank}


def _verdict(oracleValue, nativeValue, mirror):
    """``pass`` if equal, ``mirror`` if equal after applying ``mirror``, else ``mismatch``."""
    if oracleValue == nativeValue:
        return "pass"
    if mirror(oracleValue) == nativeValue:
        return "mirror"
    return "mismatch"


def _nativeValue(knot, computeName):
    from tetradrome import invariants
    return invariants.compute(knot, computeName).value


def _agreeGroups(knot, computeName, oracleGroups):
    """Verdict for a {(h, q): rank} oracle value against native, up to the Khovanov mirror."""
    return _verdict(oracleGroups, _nativeValue(knot, computeName), _mirrorKhovanov)


def _agreeScalar(knot, computeName, oracleValue, mirror=lambda v: -v):
    """Verdict for a scalar (e.g. Rasmussen s) against native; the mirror knot's value is
    ``mirror(oracleValue)`` (negation for s)."""
    return _verdict(oracleValue, _nativeValue(knot, computeName), mirror)


def _fieldAfterColon(text, marker):
    """Text after ``:`` on the first line containing ``marker``, or None."""
    for line in text.splitlines():
        if marker in line:
            return line.split(":", 1)[1].strip()
    return None


# ---- KnotJob (rational + F2 Khovanov, Rasmussen s) -----------------------------------------

def knotjobRun(knot, reps):
    """One ``knotjob -kb0 -s0`` call (PD file in, results file out) yields rational Khovanov (the
    integral free part), F2 Khovanov (that free part plus the order-2 torsion via UCT), and
    Rasmussen s. KnotJob reads Tetradrome's PD in the mirror convention, so each is judged up to
    mirror. The Khovanov call carries the measured time; the others come from the same call."""
    import os
    import subprocess
    import tempfile
    try:
        bracket = _bracketPD(knot)
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, "knot.txt"), "w") as handle:
                handle.write(bracket + "\n")

            def call():
                subprocess.run(["knotjob", "knot.txt", "-kb0", "-s0"], cwd=work,
                               check=True, capture_output=True, text=True)
                out = os.path.join(work, "knot.txt_s0_kb0")
                if not os.path.exists(out):
                    raise RuntimeError("knotjob wrote no knot.txt_s0_kb0 output file")
                with open(out) as handle:
                    return handle.read()

            text, seconds = _best(call, reps)

        freeText = _fieldAfterColon(text, "Integral unreduced Khovanov Homology")
        sText = _fieldAfterColon(text, "S-Invariant mod 0")
        if freeText is None or sText is None:
            raise ValueError("no Khovanov / s lines in knotjob output")
        torsionText = _fieldAfterColon(text, "Torsion of order 2")
        free = _parseKhovanovPoly(freeText)
        torsion = _parseKhovanovPoly(torsionText) if torsionText else {}
        f2 = _f2FromIntegral(free, torsion)
        s = int(sText)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        return {name: miss for name in
                ("rational_khovanov_homology", "khovanov_homology", "rasmussen_s")}

    return {
        "rational_khovanov_homology": Measurement(
            value=f"total_rank={sum(free.values())}", seconds=seconds, note="knotjob -kb0 -s0",
            agree=_agreeGroups(knot, "rational_khovanov_homology", free)),
        "khovanov_homology": Measurement(
            value=f"total_rank={sum(f2.values())}", seconds=None,
            note="F2 via UCT from integral + torsion; same call",
            agree=_agreeGroups(knot, "khovanov_homology", f2)),
        "rasmussen_s": Measurement(
            value=str(s), seconds=None, note="same call",
            agree=_agreeScalar(knot, "rasmussen_s", s)),
    }


# ---- JavaKh (rational Khovanov) ------------------------------------------------------------

def javakhAvailable():
    return _probeBinary("javakh", "JavaKh")


def javakhRun(knot, reps):
    """One ``javakh -Q`` call (bracket PD on stdin) -> rational Khovanov as a quoted
    ``q^a*t^b`` string. JavaKh reads Tetradrome PD in the mirror convention, judged up to mirror."""
    import subprocess
    try:
        bracket = _bracketPD(knot)

        def call():
            proc = subprocess.run(["javakh", "-Q"], input=bracket + "\n",
                                  check=True, capture_output=True, text=True)
            return proc.stdout

        out, seconds = _best(call, reps)
        groups = _parseKhovanovPoly(out.replace('"', ""))
        if not groups:
            raise ValueError("no parseable Khovanov terms in javakh output: %r" % out[:80])
    except Exception as error:
        return {"rational_khovanov_homology": Measurement(
            value=f"error: {type(error).__name__}", seconds=None, note=str(error)[:80],
            agree="n/a")}
    return {"rational_khovanov_homology": Measurement(
        value=f"total_rank={sum(groups.values())}", seconds=seconds, note="javakh -Q",
        agree=_agreeGroups(knot, "rational_khovanov_homology", groups))}


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
    Oracle("knotjob", knotjobAvailable, knotjobRun),
    Oracle("javakh", javakhAvailable, javakhRun),
    Oracle("sage", sageAvailable, None),
    Oracle("khoca", khocaAvailable, None),
]
