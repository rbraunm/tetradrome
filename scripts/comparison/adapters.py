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

import ast
import dataclasses
import hashlib
import importlib
import os
import re
import shutil
import subprocess
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
    verdict = result.validation.verdict("knotinfo")
    agree = {"pass": "pass", "not_run": "no-oracle"}.get(verdict, verdict)
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


# ---- KhoHo (rational Khovanov, (2,n) torus knots) ------------------------------------------

def khohoAvailable():
    return _probeBinary("khoho", "KhoHo")


_TORUS_2N = re.compile(r"^(\d+)_1$")


def _torusParams(identity):
    """(2, n) for the (2, n) torus knot spelled ``n_1`` with n odd >= 3 (KhoHo's ``torus`` input);
    None otherwise (a non-torus KhoHo input path is not wired yet -- Wave 2)."""
    if not identity:
        return None
    match = _TORUS_2N.match(identity)
    if not match:
        return None
    n = int(match.group(1))
    return (2, n) if n >= 3 and n % 2 == 1 else None


def _khohoPoly(text):
    """KhoHo prints gp progress, then the Khovanov polynomial on the final line. Return the last
    line that is polynomial-only (q, t, digits, exponents, +, *, parens, spaces) and mentions q."""
    poly = None
    for line in text.splitlines():
        stripped = line.strip()
        if "q" in stripped and re.fullmatch(r"[q t0-9\^\*\+\(\)\-]+", stripped):
            poly = stripped
    return poly


def khohoRun(knot, reps):
    """One ``KhPol_Q(torus(2,n))`` call via gp (KhoHo) for the (2,n) torus knots -> rational
    Khovanov. KhoHo's ``torus(2,n)`` is the positive torus knot, the mirror of KnotInfo's n_1, so
    it is judged up to mirror. Non-torus knots report n/a (no KhoHo input path wired yet)."""
    params = _torusParams(getattr(knot, "identity", None))
    if params is None:
        return {"rational_khovanov_homology": Measurement(
            value="n/a", seconds=None, note="KhoHo torus input; non-(2,n)-torus knot",
            agree="n/a")}
    import subprocess
    m, n = params
    try:
        program = "print(KhPol_Q(torus(%d,%d)));\n" % (m, n)

        def call():
            proc = subprocess.run(["khoho"], input=program,
                                  check=True, capture_output=True, text=True)
            return proc.stdout

        out, seconds = _best(call, reps)
        polyText = _khohoPoly(out)
        if polyText is None:
            raise ValueError("no Khovanov polynomial line in khoho output")
        groups = _parseKhovanovPoly(polyText)
        if not groups:
            raise ValueError("empty Khovanov polynomial from khoho: %r" % polyText)
    except Exception as error:
        return {"rational_khovanov_homology": Measurement(
            value=f"error: {type(error).__name__}", seconds=None, note=str(error)[:80],
            agree="n/a")}
    return {"rational_khovanov_homology": Measurement(
        value=f"total_rank={sum(groups.values())}", seconds=seconds,
        note="KhPol_Q(torus(%d,%d))" % (m, n),
        agree=_agreeGroups(knot, "rational_khovanov_homology", groups))}


# ---- SnapPy (hyperbolic volume) ------------------------------------------------------------

def snappyRun(knot, reps):
    """One SnapPy volume for a hyperbolic knot (by its KnotInfo name). Hyperbolic volume is not a
    native invariant, so this is oracle-only data (no agreement verdict). Non-hyperbolic knots
    (torus knots, etc.) report n/a rather than a degenerate number."""
    import snappy
    name = getattr(knot, "identity", None)
    if not name:
        return {"hyperbolic_volume": Measurement(
            value="n/a", seconds=None, note="no KnotInfo name for SnapPy", agree="n/a")}
    try:
        def call():
            manifold = snappy.Manifold(str(name))
            solution = manifold.solution_type()
            if solution != "all tetrahedra positively oriented":
                raise ValueError("non-geometric solution: %s" % solution)
            return float(manifold.volume())

        volume, seconds = _best(call, reps)
    except Exception as error:
        return {"hyperbolic_volume": Measurement(
            value="n/a", seconds=None,
            note="not hyperbolic (%s)" % type(error).__name__, agree="n/a")}
    return {"hyperbolic_volume": Measurement(
        value=f"{volume:.10f}", seconds=seconds, note="snappy Manifold(name).volume()",
        agree="oracle")}


# ---- shared classical-polynomial normalization (single-variable Laurent) -------------------

def _splitLaurentTerms(text):
    """Split a single-variable Laurent polynomial into signed terms, treating ``+``/``-`` as term
    separators except when they are an exponent sign (immediately after ``^``)."""
    text = text.replace(" ", "")
    terms, current = [], ""
    for i, ch in enumerate(text):
        if ch in "+-" and i > 0 and text[i - 1] != "^":
            terms.append(current)
            current = ch
        else:
            current += ch
    if current:
        terms.append(current)
    return terms


def _parseLaurentTerm(term, var):
    """A single ``[sign][coeff]var^exp`` (or a bare constant) -> (exponent, coefficient)."""
    term = term.replace("*", "")
    sign = 1
    if term[:1] == "+":
        term = term[1:]
    elif term[:1] == "-":
        sign, term = -1, term[1:]
    if var in term:
        left, _, right = term.partition(var)
        coeff = int(left) if left else 1
        exponent = int(right[1:]) if right.startswith("^") else 1
        return exponent, sign * coeff
    return 0, sign * int(term)


def _parseLaurent(text, var):
    """Single-variable Laurent polynomial (e.g. regina's Jones in x) -> {exponent: coefficient},
    zero coefficients dropped. Handles signs, negative exponents, an implicit exponent 1, and a
    bare constant term."""
    poly: dict = {}
    for term in _splitLaurentTerms(text):
        if not term or term in "+-":
            continue
        exponent, coeff = _parseLaurentTerm(term, var)
        poly[exponent] = poly.get(exponent, 0) + coeff
    return {e: c for e, c in poly.items() if c}


def _negateExponents(poly):
    """The t <-> t^-1 Jones convention flip: {e: c} -> {-e: c}."""
    return {-e: c for e, c in poly.items()}


def _nativeJonesDict(knot):
    """Native Jones (low, coeffs ascending in t) -> {t-exponent: coefficient}."""
    low, coeffs = _nativeValue(knot, "jones_polynomial")
    return {low + i: c for i, c in enumerate(coeffs) if c}


def _agreeJones(knot, oracleJones):
    """Verdict for a {t-exponent: coeff} Jones against native, up to the t <-> t^-1 convention."""
    return _verdict(oracleJones, _nativeJonesDict(knot), _negateExponents)


# ---- regina (Jones, HOMFLY) ----------------------------------------------------------------

def reginaAvailable():
    return _probeImport("regina", "Regina")


def reginaRun(knot, reps):
    """regina's Jones (a Laurent poly in x = t^(1/2)) mapped to native's t by halving exponents,
    judged up to the t <-> t^-1 convention (regina matches native directly). HOMFLY is reported as
    oracle-only data -- native computes no HOMFLY. regina reads the PD via Link.fromPD."""
    import regina
    try:
        pd = pdAsList(knot)

        def call():
            link = regina.Link.fromPD(pd)
            return str(link.jones()), str(link.homfly())

        (jonesText, homflyText), seconds = _best(call, reps)
        xPoly = _parseLaurent(jonesText, "x")
        if not xPoly or any(e % 2 for e in xPoly):
            raise ValueError("unexpected regina Jones (x = t^1/2): %r" % jonesText)
        jones = {e // 2: c for e, c in xPoly.items()}
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        return {"jones_polynomial": miss, "homfly_polynomial": miss}
    return {
        "jones_polynomial": Measurement(
            value=_shortValue(jones), seconds=seconds, note="regina jones(); x = t^1/2",
            agree=_agreeJones(knot, jones)),
        "homfly_polynomial": Measurement(
            value=homflyText.strip()[:40] or "?", seconds=None,
            note="regina homfly(); oracle-only", agree="oracle"),
    }


# ---- SageMath (Jones, Alexander, determinant, signature, Khovanov) --------------------------
#
# Sage runs once per knot and prints structured data (coefficient dicts, invariant-factor tuples,
# ints), so there is nothing fragile to parse. Sage reads Tetradrome PD in native's own
# convention: Khovanov and determinant match directly, Alexander matches after re-canonicalizing
# (it is defined only up to a unit +/- t^k), while Jones and signature come back in the opposite
# convention (t <-> t^-1 for Jones, a sign flip for signature) and so read as "mirror".

_SAGE_SCRIPT = """L = Link(%(pd)s)
J = L.jones_polynomial()
print("JONES", {int(e): int(c) for c, e in J.coefficients()})
A = L.alexander_polynomial()
print("ALEXANDER", {int(k): int(v) for k, v in A.dict().items()})
print("SIGNATURE", int(L.signature()))
print("DETERMINANT", int(L.determinant()))
K = L.khovanov_homology()
kh = {}
for q in K:
    for h in K[q]:
        inv = tuple(int(x) for x in K[q][h].invariants())
        if inv:
            kh[(int(h), int(q))] = inv
print("KHOVANOV", kh)
"""

_SAGE_TAGS = ("JONES", "ALEXANDER", "SIGNATURE", "DETERMINANT", "KHOVANOV")


def _parseSageFields(text):
    """Pull the tagged structured lines the sage script prints into {tag: value}."""
    fields = {}
    for line in text.splitlines():
        for tag in _SAGE_TAGS:
            if line.startswith(tag + " "):
                fields[tag] = ast.literal_eval(line[len(tag) + 1:])
    return fields


def _sageKhovanov(cells):
    """{(h, q): invariant-factor tuple} (0 = a Z summand) -> (free ranks, order-2 torsion counts);
    even-order factors are the ones that survive to F2."""
    free, torsion = {}, {}
    for (h, q), invariants in cells.items():
        rank = sum(1 for x in invariants if x == 0)
        even = sum(1 for x in invariants if x != 0 and x % 2 == 0)
        if rank:
            free[(h, q)] = rank
        if even:
            torsion[(h, q)] = even
    return free, torsion


def _nativeAlexanderDict(knot):
    """Native Alexander (ascending coeffs, lowest term at t^0) -> {exponent: coefficient}."""
    coeffs = _nativeValue(knot, "alexander_polynomial")
    return {i: c for i, c in enumerate(coeffs) if c}


def _canonicalAlexander(poly):
    """Put a Laurent Alexander polynomial in native's canonical form: shift the lowest nonzero term
    to t^0, then flip the sign so the constant term is positive (Alexander is defined up to a unit
    +/- t^k)."""
    if not poly:
        return {}
    low = min(poly)
    shifted = {e - low: c for e, c in poly.items()}
    if shifted[min(shifted)] < 0:
        shifted = {e: -c for e, c in shifted.items()}
    return shifted


def _agreeAlexander(knot, oracleAlexander):
    """Verdict for an Alexander polynomial against native, up to the +/- t^k unit."""
    return "pass" if _canonicalAlexander(oracleAlexander) == _nativeAlexanderDict(knot) else "mismatch"


def sageRun(knot, reps):
    """One sage run per knot -> Jones, Alexander, determinant, signature, and rational + F2
    Khovanov (F2 via UCT from the invariant factors). Sage shares native's PD convention, so
    Khovanov and determinant are direct, Alexander is up to canonicalization, and Jones/signature
    read as mirror (opposite variable / sign convention). One call carries the time."""
    import os
    import subprocess
    import tempfile
    names = ("jones_polynomial", "alexander_polynomial", "determinant", "signature",
             "rational_khovanov_homology", "khovanov_homology")
    try:
        script = _SAGE_SCRIPT % {"pd": repr(pdAsList(knot))}
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "compute.sage")
            with open(path, "w") as handle:
                handle.write(script)

            def call():
                proc = subprocess.run(["sage", path], check=True, capture_output=True, text=True)
                return proc.stdout

            out, seconds = _best(call, reps)
        fields = _parseSageFields(out)
        missing = [tag for tag in _SAGE_TAGS if tag not in fields]
        if missing:
            raise ValueError("sage output missing %s" % ", ".join(missing))
        jones = {int(e): int(c) for e, c in fields["JONES"].items()}
        alexander = {int(e): int(c) for e, c in fields["ALEXANDER"].items()}
        signature = int(fields["SIGNATURE"])
        determinant = int(fields["DETERMINANT"])
        free, torsion = _sageKhovanov(fields["KHOVANOV"])
        f2 = _f2FromIntegral(free, torsion)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        return {name: miss for name in names}
    return {
        "jones_polynomial": Measurement(
            value=_shortValue(jones), seconds=seconds, note="sage jones_polynomial()",
            agree=_agreeJones(knot, jones)),
        "alexander_polynomial": Measurement(
            value=_shortValue(alexander), seconds=None,
            note="sage alexander_polynomial(); same run", agree=_agreeAlexander(knot, alexander)),
        "determinant": Measurement(
            value=str(determinant), seconds=None, note="same run",
            agree=_agreeScalar(knot, "determinant", determinant, mirror=lambda v: v)),
        "signature": Measurement(
            value=str(signature), seconds=None, note="same run",
            agree=_agreeScalar(knot, "signature", signature)),
        "rational_khovanov_homology": Measurement(
            value=f"total_rank={sum(free.values())}", seconds=None,
            note="sage khovanov_homology(); same run",
            agree=_agreeGroups(knot, "rational_khovanov_homology", free)),
        "khovanov_homology": Measurement(
            value=f"total_rank={sum(f2.values())}", seconds=None,
            note="F2 via UCT from invariant factors; same run",
            agree=_agreeGroups(knot, "khovanov_homology", f2)),
    }


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


# ---- oracle versions (ADR 0013) -----------------------------------------------------------
# Each probe mirrors what scripts/install_oracles.sh records, so the artifact's versions match the
# provisioned host exactly: pip distributions by metadata, source oracles by their built git sha,
# the rolling KnotJob jar by content hash. An absent oracle reports "absent" (the generator only
# records versions for oracles it also found present).

def _oracleHome():
    return os.environ.get("ORACLE_HOME", "/opt/oracles")


def _pipVersion(distribution):
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as packageVersion
    try:
        return packageVersion(distribution)
    except PackageNotFoundError:
        return "absent"


def _gitShaVersion(subdir):
    path = os.path.join(_oracleHome(), subdir)
    done = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    return f"git:{done.stdout.strip()}" if done.returncode == 0 and done.stdout.strip() else "absent"


def _jarHashVersion(relativePath):
    path = os.path.join(_oracleHome(), relativePath)
    try:
        with open(path, "rb") as handle:
            return f"sha256:{hashlib.sha256(handle.read()).hexdigest()[:12]}"
    except OSError:
        return "absent"


def kfhVersion():
    return _pipVersion("knot_floer_homology")


def snappyVersion():
    return _pipVersion("snappy")


def reginaVersion():
    return _pipVersion("regina")


def khocaVersion():
    return _pipVersion("khoca")


def knotjobVersion():
    return _jarHashVersion("knotjob/KnotJob/KnotJob.jar")


def javakhVersion():
    return _gitShaVersion("javakh")


def khohoVersion():
    return _gitShaVersion("khoho")


def sageVersion():
    exe = shutil.which("sage")
    if not exe:
        return "absent"
    done = subprocess.run([exe, "--version"], capture_output=True, text=True)
    line = (done.stdout.strip().splitlines() or [""])[0]
    marker = "SageMath version "
    return line.split(marker, 1)[1].split(",")[0].strip() if marker in line else "present"


# Registry the generator iterates. ``run`` is None for probe-only oracles -- the generator prints
# "adapter pending" for the invariants they cover until a run is wired against a real install.
# ``version`` mirrors what install_oracles.sh recorded for the oracle (ADR 0013).
@dataclasses.dataclass
class Oracle:
    key: str
    available: object           # () -> (bool, detail)
    run: object | None          # (knot, reps) -> {invariantName: Measurement} | None
    version: object             # () -> version string, or "absent"


ORACLES = [
    Oracle("kfh", kfhAvailable, kfhRun, kfhVersion),
    Oracle("snappy", snappyAvailable, snappyRun, snappyVersion),
    Oracle("regina", reginaAvailable, reginaRun, reginaVersion),
    Oracle("knotjob", knotjobAvailable, knotjobRun, knotjobVersion),
    Oracle("javakh", javakhAvailable, javakhRun, javakhVersion),
    Oracle("khoho", khohoAvailable, khohoRun, khohoVersion),
    Oracle("sage", sageAvailable, sageRun, sageVersion),
    Oracle("khoca", khocaAvailable, None, khocaVersion),
]
