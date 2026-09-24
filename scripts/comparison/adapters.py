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
    """Time the native computation and capture its validation verdict -- separately.

    The timed call runs validate="off" so the artifact's native timing claim contains no
    oracle time (a knotjob subprocess inside the timed lambda would skew a 10ms Khovanov
    cell by 20x); the verdicts come from one untimed soft call, and the two values must
    agree or this raises. An invariant whose wired computed oracles are ALL missing from
    this host is a provisioning failure of the artifact run itself, not a per-cell state --
    fail loud before timing anything (ADR 0004). A partially-provisioned invariant is fine:
    strict's own rule is >= 1 computed pass, so regina alone validates a classical cell on a
    sage-less host. Soft mode then tolerates only the honestly-recorded gaps. The agree
    verdict prefers a computed oracle's pass over KnotInfo's.
    """
    from tetradrome import invariants
    from tetradrome.backends import registry
    wired = registry.wired_validators(computeName)
    if wired and not any(v.is_available() for v in wired):
        missing = ", ".join(v.name for v in wired)
        raise RuntimeError(
            f"{computeName}: no wired computed oracle is installed on this host "
            f"({missing} all missing) -- run scripts/install_oracles.sh before "
            f"generating the artifact."
        )
    try:
        result, seconds = _best(lambda: invariants.compute(knot, computeName, validate="off"), reps)
        # Verdicts come from ONE untimed soft call, so oracle consultations (e.g. the
        # knotjob subprocess) never contaminate the native timing the artifact claims.
        validated = invariants.compute(knot, computeName, validate="soft")
    except Exception as error:                       # an engine that cannot run on this knot
        return Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:60], agree="n/a")
    if validated.value != result.value:
        raise RuntimeError(f"{computeName}: nondeterministic native value across runs")
    if any(v.verdict == "pass" for v in validated.validation.validators if v.oracle != "knotinfo"):
        agree = "pass"
    else:
        verdict = validated.validation.verdict("knotinfo")
        agree = {"pass": "pass", "not_run": "no-oracle"}.get(verdict, verdict)
    return Measurement(value=_shortValue(validated.value), seconds=seconds, agree=agree)


def _shortValue(value):
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."



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
    """One timed kfh call, delegated to the src ``hfk_adapter.raw_hfk`` (the single caller of
    the library); returns {invariantName: Measurement}. The HFK ranks carry the measured time;
    the scalar invariants come from the SAME call and are noted as such."""
    from tetradrome.backends import hfk_adapter
    try:
        out, seconds = _best(lambda: hfk_adapter.raw_hfk(knot), reps)
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
    """The native value as an untimed reference for judging an oracle's output. Validation
    of the native value itself is the tetra cell's job, so no validators are consulted here
    (they would only add oracle subprocesses to every agreement judgment)."""
    from tetradrome import invariants
    return invariants.compute(knot, computeName, validate="off").value


def _agreeGroups(knot, computeName, oracleGroups):
    """Verdict for a {(h, q): rank} oracle value against native, up to the Khovanov mirror."""
    return _verdict(oracleGroups, _nativeValue(knot, computeName), _mirrorKhovanov)


def _agreeScalar(knot, computeName, oracleValue, mirror=lambda v: -v):
    """Verdict for a scalar (e.g. Rasmussen s) against native; the mirror knot's value is
    ``mirror(oracleValue)`` (negation for s)."""
    return _verdict(oracleValue, _nativeValue(knot, computeName), mirror)


# ---- KnotJob (Khovanov family, Rasmussen s, sl(3)) -----------------------------------------

_KNOTJOB_TORSION = re.compile(r"^Torsion of order (\d+)$")


def _knotjobSections(text):
    """KnotJob output -> ({heading: {"free": groups, "torsion": {order: groups}}}, {scalar: str}).

    A ``<heading> : <polynomial>`` line whose heading names a Homology opens a section, and every
    ``Torsion of order N : <polynomial>`` line after it belongs to that section -- KnotJob prints
    torsion per section and per order, so reading only the first ``Torsion of order 2`` line would
    silently drop reduced-section torsion and any 2^k-torsion. Other ``name : value`` lines
    (``S-Invariant mod 0``) are scalars; the ``Knot 1`` label carries no colon. A torsion line
    with no section above it raises."""
    sections, scalars, current = {}, {}, None
    for line in (raw.strip() for raw in text.splitlines()):
        if ":" not in line:
            continue
        name, value = (part.strip() for part in line.split(":", 1))
        torsion = _KNOTJOB_TORSION.match(name)
        if torsion:
            if current is None:
                raise ValueError(f"knotjob torsion line before any homology: {line!r}")
            sections[current]["torsion"][int(torsion.group(1))] = _parseKhovanovPoly(value)
        elif "Homology" in name:
            current = name
            sections[current] = {"free": _parseKhovanovPoly(value), "torsion": {}}
        else:
            scalars[name] = value
    return sections, scalars


def _knotjobSection(sections, heading):
    if heading not in sections:
        raise ValueError(f"no {heading!r} in knotjob output (have {sorted(sections)})")
    return sections[heading]


def _evenTorsion(section):
    """Every torsion summand of even order, merged: each Z/N with N even contributes to F2 by the
    universal coefficient theorem (Z/N (x) F2 = Tor(Z/N, F2) = F2), and odd orders contribute
    nothing."""
    merged = {}
    for order, groups in section["torsion"].items():
        if order % 2 == 0:
            for key, count in groups.items():
                merged[key] = merged.get(key, 0) + count
    return merged


def _integralSummary(section):
    """``free=R`` plus ``Z/N x k`` per torsion order -- a mirror-independent summary for cells
    with no native value to compare against."""
    parts = [f"free={sum(section['free'].values())}"]
    for order in sorted(section["torsion"]):
        parts.append(f"Z/{order}x{sum(section['torsion'][order].values())}")
    return " ".join(parts)


def _khovanovWidth(section):
    """Homological width: the number of diagonals delta = q - 2h carrying any generator, free or
    torsion. For a knot every delta has the same parity, so width = (max - min) / 2 + 1; a
    mixed-parity support means the groups were mis-parsed, and raises."""
    keys = set(section["free"])
    for groups in section["torsion"].values():
        keys |= set(groups)
    if not keys:
        raise ValueError("no Khovanov support to take the width of")
    deltas = [q - 2 * h for h, q in keys]
    span = max(deltas) - min(deltas)
    if span % 2:
        raise ValueError(f"Khovanov support spans mixed-parity diagonals: {sorted(set(deltas))}")
    return span // 2 + 1


def _knotjobCall(bracket, flags, outputName, reps):
    """One timed knotjob run in a fresh scratch directory, returning (output text, seconds).
    Each rep gets its own directory because ``-ks`` writes its result over the input file."""
    import os
    import tempfile

    def call():
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, "knot.txt"), "w") as handle:
                handle.write(bracket + "\n")
            subprocess.run(["knotjob", "knot.txt", *flags], cwd=work, check=True,
                           capture_output=True, text=True)
            out = os.path.join(work, outputName)
            with open(out) as handle:
                text = handle.read()
            if "Homology" not in text:
                raise RuntimeError(f"knotjob {' '.join(flags)} wrote no homology to {outputName}")
            return text

    return _best(call, reps)


def _knotjobMiss(names, error):
    miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                       note=str(error)[:80], agree="n/a")
    return {name: miss for name in names}


def knotjobRun(knot, reps):
    """Three KnotJob runs, each timed. KnotJob reads Tetradrome's PD in the mirror convention.

    ``-kb0 -s0`` yields rational Khovanov (the integral free part), F2 Khovanov (that free part
    plus all even-order torsion via UCT), and Rasmussen s -- native rows, judged up to mirror --
    plus integral Khovanov with torsion, the reduced theory, and the homological width, which
    have no native engine yet and so are oracle-only cells. ``-ko0`` is odd Khovanov and
    ``-ks0`` is sl(3) homology, both oracle-only. The -kb0 call carries its measured time; the
    other rows from it say "same call"."""
    rows = {}
    bracket = _bracketPD(knot)
    evenNames = ("rational_khovanov_homology", "khovanov_homology", "rasmussen_s",
                 "khovanov_integral", "khovanov_reduced", "khovanov_width")
    try:
        text, seconds = _knotjobCall(bracket, ["-kb0", "-s0"], "knot.txt_s0_kb0", reps)
        sections, scalars = _knotjobSections(text)
        unreduced = _knotjobSection(sections, "Integral unreduced Khovanov Homology")
        reduced = _knotjobSection(sections, "Integral reduced Khovanov Homology")
        if "S-Invariant mod 0" not in scalars:
            raise ValueError("no S-Invariant mod 0 line in knotjob output")
        s = int(scalars["S-Invariant mod 0"])
        free = unreduced["free"]
        f2 = _f2FromIntegral(free, _evenTorsion(unreduced))
        width = _khovanovWidth(unreduced)
    except Exception as error:
        rows.update(_knotjobMiss(evenNames, error))
    else:
        same = "same knotjob -kb0 -s0 call"
        rows.update({
            "rational_khovanov_homology": Measurement(
                value=f"total_rank={sum(free.values())}", seconds=seconds,
                note="knotjob -kb0 -s0",
                agree=_agreeGroups(knot, "rational_khovanov_homology", free)),
            "khovanov_homology": Measurement(
                value=f"total_rank={sum(f2.values())}", seconds=None,
                note=f"{same}; F2 via UCT from all even-order torsion",
                agree=_agreeGroups(knot, "khovanov_homology", f2)),
            "rasmussen_s": Measurement(
                value=f"s={s}", seconds=None, note=same,
                agree=_agreeScalar(knot, "rasmussen_s", s)),
            "khovanov_integral": Measurement(
                value=_integralSummary(unreduced), seconds=None, note=same, agree="oracle"),
            "khovanov_reduced": Measurement(
                value=_integralSummary(reduced), seconds=None, note=f"{same}; integral",
                agree="oracle"),
            "khovanov_width": Measurement(
                value=f"width={width}", seconds=None,
                note=f"{same}; diagonals of integral Khovanov, torsion included", agree="oracle"),
        })
    try:
        text, seconds = _knotjobCall(bracket, ["-ko0"], "knot.txt_ko0", reps)
        odd = _knotjobSection(_knotjobSections(text)[0], "Odd integral Khovanov Homology")
    except Exception as error:
        rows.update(_knotjobMiss(("khovanov_odd",), error))
    else:
        rows["khovanov_odd"] = Measurement(
            value=_integralSummary(odd), seconds=seconds, note="knotjob -ko0; integral",
            agree="oracle")
    try:
        text, seconds = _knotjobCall(bracket, ["-ks0"], "knot.txt", reps)
        sl3 = _knotjobSection(_knotjobSections(text)[0], "Unreduced integral sl_3 Homology")
    except Exception as error:
        rows.update(_knotjobMiss(("sl_n_homology",), error))
    else:
        rows["sl_n_homology"] = Measurement(
            value=_integralSummary(sl3), seconds=seconds,
            note="knotjob -ks0; N = 3, unreduced, integral", agree="oracle")
    return rows


# ---- JavaKh (rational Khovanov) ------------------------------------------------------------

def javakhAvailable():
    return _probeBinary("javakh", "JavaKh")


_JAVAKH_TERM = re.compile(r"^(?:(\d+)\*)?q\^(-?\d+)\*t\^(-?\d+)\*Z\[([\d,]+)\]$")


def _parseJavakhIntegral(text):
    """``javakh -Z`` output -> {"free": groups, "torsion": {order: groups}}, RAW convention.

    Each term is ``q^a*t^b*Z[o1,o2,...]`` with one entry per cyclic summand at that bidegree:
    0 is a free Z and n is Z/n, so ``Z[0,0,2]`` is Z^2 + Z/2. An optional leading ``c*``
    multiplies the term. Any term that does not match raises rather than being skipped."""
    body = text.strip().strip('"').strip()
    if not body:
        raise ValueError("empty javakh -Z output")
    free, torsion = {}, {}
    for term in (piece.strip() for piece in body.split(" + ")):
        match = _JAVAKH_TERM.match(term)
        if not match:
            raise ValueError(f"unparseable javakh -Z term: {term!r}")
        count = int(match.group(1) or 1)
        key = (int(match.group(3)), int(match.group(2)))
        for order in (int(entry) for entry in match.group(4).split(",")):
            bucket = free if order == 0 else torsion.setdefault(order, {})
            bucket[key] = bucket.get(key, 0) + count
    return {"free": free, "torsion": torsion}


def _javakh(flag, bracket):
    return subprocess.run(["javakh", flag], input=bracket + "\n",
                          check=True, capture_output=True, text=True).stdout


def javakhRun(knot, reps):
    """Two JavaKh calls on the bracket PD (stdin), each timed. JavaKh reads Tetradrome PD in the
    mirror convention.

    ``-Q`` gives rational Khovanov as a quoted ``q^a*t^b`` string, judged up to mirror.
    ``-Z`` gives integral Khovanov with torsion, which fills three more rows. F2 comes from it by
    UCT applied to the RAW groups and then mirrored -- that order reproduces native F2 on 3_1,
    4_1, 5_2, 6_1, 7_4, 8_19 and 10_124, while mirroring first fails on all seven, because
    mirroring integral homology also moves torsion one degree. Integral and width are
    oracle-only cells; width is taken on the raw groups, which is sound since width is
    mirror-invariant. JavaKh has no reduced theory."""
    rows = {}
    bracket = _bracketPD(knot)
    try:
        out, seconds = _best(lambda: _javakh("-Q", bracket), reps)
        groups = _parseKhovanovPoly(out.replace('"', ""))
        if not groups:
            raise ValueError("no parseable Khovanov terms in javakh output: %r" % out[:80])
    except Exception as error:
        rows["rational_khovanov_homology"] = Measurement(
            value=f"error: {type(error).__name__}", seconds=None, note=str(error)[:80],
            agree="n/a")
    else:
        rows["rational_khovanov_homology"] = Measurement(
            value=f"total_rank={sum(groups.values())}", seconds=seconds, note="javakh -Q",
            agree=_agreeGroups(knot, "rational_khovanov_homology", groups))
    try:
        integral, seconds = _best(lambda: _parseJavakhIntegral(_javakh("-Z", bracket)), reps)
        f2 = _mirrorKhovanov(_f2FromIntegral(integral["free"], _evenTorsion(integral)))
        width = _khovanovWidth(integral)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        rows.update({name: miss for name in
                     ("khovanov_homology", "khovanov_integral", "khovanov_width")})
    else:
        same = "same javakh -Z call"
        rows.update({
            "khovanov_integral": Measurement(
                value=_integralSummary(integral), seconds=seconds, note="javakh -Z",
                agree="oracle"),
            "khovanov_homology": Measurement(
                value=f"total_rank={sum(f2.values())}", seconds=None,
                note=f"{same}; F2 via UCT on the raw groups, then mirrored",
                agree=_agreeGroups(knot, "khovanov_homology", f2)),
            "khovanov_width": Measurement(
                value=f"width={width}", seconds=None,
                note=f"{same}; diagonals of integral Khovanov, torsion included",
                agree="oracle"),
        })
    return rows


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


# ---- Khoca (rational + F2 Khovanov, natively per coefficient ring) -------------------------

def _khocaGroups(pd, ring):
    """Unreduced sl(2) homology over a FIELD coefficient ring (1 = Q, 2 = F2) -> {(h, q): dim}
    in KnotInfo's q-convention. Khoca's own quantum grading is negated relative to KnotInfo's
    (verified empirically on 3_1/4_1/5_2/8_19/10_124: q-negation matched 10/10 across both
    rings; direct and full mirror matched 0/10), so q is negated here as a fixed, documented
    normalization. Rows are [t, q, torsionOrder, multiplicity] and the output is
    [reduced, unreduced]; a field ring must never produce a torsion row or a negative
    aggregate, and both fail loud."""
    import khoca
    out = khoca.InteractiveCalculator(coefficient_ring=ring)(pd)
    groups = {}
    for t, q, torsion, multiplicity in out[1]:
        if torsion != 0:
            raise ValueError(f"field ring produced torsion row {(t, q, torsion, multiplicity)}")
        key = (t, -q)
        groups[key] = groups.get(key, 0) + multiplicity
    groups = {key: rank for key, rank in groups.items() if rank}
    if any(rank < 0 for rank in groups.values()):
        raise ValueError(f"negative aggregate rank in khoca output: {groups}")
    return groups


def _khocaIntegralSection(rows):
    """One half of a ring-0 (integral) khoca result -> {"free": groups, "torsion": {order: groups}}
    in the canonical convention. Rows are ``[t, q, torsionOrder, multiplicity]`` and may carry
    zero or negative multiplicities that cancel per key, so they are aggregated before use and a
    negative aggregate raises. The q-negation is khoca's verified transform. The torsion needs no
    further shift: F2 derived from these groups by UCT reproduces native F2 exactly (checked on
    3_1, 4_1, 5_2, 6_1, 7_4, 8_19, 10_124), whereas moving torsion one homological degree fails
    on all of them."""
    free, torsion = {}, {}
    for t, q, order, multiplicity in rows:
        bucket = free if order == 0 else torsion.setdefault(order, {})
        bucket[(t, -q)] = bucket.get((t, -q), 0) + multiplicity
    free = {key: rank for key, rank in free.items() if rank}
    torsion = {order: {key: count for key, count in groups.items() if count}
               for order, groups in torsion.items()}
    torsion = {order: groups for order, groups in torsion.items() if groups}
    for groups in (free, *torsion.values()):
        if any(count < 0 for count in groups.values()):
            raise ValueError(f"negative aggregate multiplicity in khoca integral output: {groups}")
    return {"free": free, "torsion": torsion}


def khocaRun(knot, reps):
    """Khovanov over Q (coefficient ring 1) and over F2 (ring 2), each computed natively by
    khoca from the PD -- no UCT derivation, unlike the KnotJob path -- plus one integral call
    (ring 0) whose unreduced and reduced halves fill the integral, reduced and width target
    rows. Each call carries its own measured time. Values are normalized to KnotInfo's
    q-convention, so agreement on the native rows is judged direct rather than up to mirror.

    sl(N) for N > 2 is not measured: khoca's own documentation says PD and braid input are
    valid only for sl(2) and give nonsensical output otherwise -- sl(N) needs a bipartite knot
    as a matched diagram, which Tetradrome does not produce. The cell says so rather than
    showing a number. KnotJob measures sl(3) from the PD."""
    rows = {}
    try:
        pd = pdAsList(knot)
        rational, secondsQ = _best(lambda: _khocaGroups(pd, 1), reps)
        f2, secondsF2 = _best(lambda: _khocaGroups(pd, 2), reps)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        rows.update({name: miss for name in
                     ("rational_khovanov_homology", "khovanov_homology")})
    else:
        rows.update({
            "rational_khovanov_homology": Measurement(
                value=f"total_rank={sum(rational.values())}", seconds=secondsQ,
                note="khoca ring Q; q-grading normalized",
                agree=_agreeGroups(knot, "rational_khovanov_homology", rational)),
            "khovanov_homology": Measurement(
                value=f"total_rank={sum(f2.values())}", seconds=secondsF2,
                note="khoca ring F2 (native, no UCT); q-grading normalized",
                agree=_agreeGroups(knot, "khovanov_homology", f2)),
        })
    try:
        import khoca
        pd = pdAsList(knot)
        out, secondsZ = _best(lambda: khoca.InteractiveCalculator(coefficient_ring=0)(pd), reps)
        unreduced = _khocaIntegralSection(out[1])
        reduced = _khocaIntegralSection(out[0])
        width = _khovanovWidth(unreduced)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        rows.update({name: miss for name in
                     ("khovanov_integral", "khovanov_reduced", "khovanov_width")})
    else:
        same = "same khoca ring-0 call"
        rows.update({
            "khovanov_integral": Measurement(
                value=_integralSummary(unreduced), seconds=secondsZ, note="khoca ring Z",
                agree="oracle"),
            "khovanov_reduced": Measurement(
                value=_integralSummary(reduced), seconds=None, note=f"{same}; integral",
                agree="oracle"),
            "khovanov_width": Measurement(
                value=f"width={width}", seconds=None,
                note=f"{same}; diagonals of integral Khovanov, torsion included",
                agree="oracle"),
        })
    rows["sl_n_homology"] = Measurement(
        value="n/a", seconds=None,
        note="khoca sl(N>2) needs a bipartite matched diagram; PD input is sl(2)-only",
        agree="n/a")
    return rows


# ---- knotkit (Rasmussen s, Khovanov over F2 and Q) ------------------------------------------

_KNOTKIT_AXIS_T = re.compile(r"\\draw \(([-\d.]+),-\.2\) node\[below\] \{\$(-?\d+)\$\};")
_KNOTKIT_AXIS_Q = re.compile(r"\\draw \(-\.2,([-\d.]+)\) node\[left\] \{\$(-?\d+)\$\};")
_KNOTKIT_FILL = re.compile(r"\\fill \(([-\d.]+), ([-\d.]+)\) circle \(\.15\);")
_KNOTKIT_NODE = re.compile(r"\\draw \(([-\d.]+), ([-\d.]+)\) node \{\$(\d+)\$\};")
_KNOTKIT_FURNITURE = re.compile(
    r"\\draw\[->\]|\\draw\[step=|\\draw \([-\d.]+,-0\.8\) node\[below\]")
_KNOTKIT_RANK = re.compile(r"\\rank Kh = (\d+)")
_KNOTKIT_S = re.compile(r"^s\(.*\) = (-?\d+)$")


def _parseKnotkitGrid(latex):
    """``kk kh`` stdout -- a standalone LaTeX document -> {(t, q): rank}, RAW convention.

    The TikZ grid's axis labels map cell centres to gradings; a generator is a filled circle
    (rank 1) or a ``node {$N$}`` (rank N) at its cell centre. Every line of the picture must be
    one of those or known furniture, and the parsed ranks must sum to kk's own ``\rank Kh``
    total: an unfamiliar drawing element raises rather than silently dropping generators."""
    if r"\begin{tikzpicture}" not in latex:
        raise ValueError("no tikzpicture in kk kh output")
    body = latex.split(r"\begin{tikzpicture}", 1)[1].split("\n", 1)[1]
    body = body.split(r"\end{tikzpicture}", 1)[0]
    tAt, qAt, cells = {}, {}, []
    for line in (raw.strip() for raw in body.splitlines()):
        if not line:
            continue
        axisT = _KNOTKIT_AXIS_T.fullmatch(line)
        axisQ = _KNOTKIT_AXIS_Q.fullmatch(line)
        fill = _KNOTKIT_FILL.fullmatch(line)
        node = _KNOTKIT_NODE.fullmatch(line)
        if axisT:
            tAt[axisT.group(1)] = int(axisT.group(2))
        elif axisQ:
            qAt[axisQ.group(1)] = int(axisQ.group(2))
        elif fill:
            cells.append((fill.group(1), fill.group(2), 1))
        elif node:
            cells.append((node.group(1), node.group(2), int(node.group(3))))
        elif not _KNOTKIT_FURNITURE.search(line):
            raise ValueError(f"unrecognised element in kk kh grid: {line!r}")
    groups = {}
    for x, y, rank in cells:
        if x not in tAt or y not in qAt:
            raise ValueError(f"kk kh generator at ({x}, {y}) has no axis label")
        key = (tAt[x], qAt[y])
        groups[key] = groups.get(key, 0) + rank
    total = _KNOTKIT_RANK.search(latex)
    if not total:
        raise ValueError(r"no \rank Kh line in kk kh output")
    if sum(groups.values()) != int(total.group(1)):
        raise ValueError(f"kk kh grid sums to {sum(groups.values())}, "
                         f"but kk reports rank {total.group(1)}")
    if not groups:
        raise ValueError("empty Khovanov homology from kk (never zero for a knot)")
    return groups


def _parseKnotkitS(text):
    """``kk s`` stdout -> s, RAW convention. Exactly one ``s(...) = n`` line or it raises."""
    matches = [_KNOTKIT_S.match(line.strip()) for line in text.splitlines()]
    values = [int(match.group(1)) for match in matches if match]
    if len(values) != 1:
        raise ValueError(f"expected one s line in kk s output, found {len(values)}: {text[:80]!r}")
    return values[0]


def _knotkit(arguments):
    return subprocess.run(["kk", *arguments], check=True, capture_output=True, text=True).stdout


def knotkitRun(knot, reps):
    """Rasmussen s over Q, and Khovanov over F2 and over Q, each one ``kk`` call on the bracket
    PD with its own measured time. Conventions verified on the chiral sweep before wiring (see
    roadmap/research/knotkit.md): s is negated relative to native (10/10, both fields) and
    Khovanov is the full mirror (h, q) -> (-h, -q) (14/14, both fields, rank > 1 cells
    included). The transforms are applied here, so agreement is judged direct -- this is a
    genuine mirror, unlike khoca's q-negation. s uses Q because native s is Rasmussen's
    original, read off Lee homology over Q; kk's own default field is Z2."""
    try:
        bracket = _bracketPD(knot)
        rawS, secondsS = _best(lambda: _parseKnotkitS(_knotkit(["s", "-f", "Q", bracket])), reps)
        rawQ, secondsQ = _best(lambda: _parseKnotkitGrid(_knotkit(["kh", "-f", "Q", bracket])),
                               reps)
        rawF2, secondsF2 = _best(lambda: _parseKnotkitGrid(_knotkit(["kh", "-f", "Z2", bracket])),
                                 reps)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        return {name: miss for name in
                ("rasmussen_s", "rational_khovanov_homology", "khovanov_homology")}
    s = -rawS
    rational = _mirrorKhovanov(rawQ)
    f2 = _mirrorKhovanov(rawF2)
    return {
        "rasmussen_s": Measurement(
            value=f"s={s}", seconds=secondsS, note="kk s -f Q; negated to canonical",
            agree=_agreeScalar(knot, "rasmussen_s", s)),
        "rational_khovanov_homology": Measurement(
            value=f"total_rank={sum(rational.values())}", seconds=secondsQ,
            note="kk kh -f Q; mirrored to canonical",
            agree=_agreeGroups(knot, "rational_khovanov_homology", rational)),
        "khovanov_homology": Measurement(
            value=f"total_rank={sum(f2.values())}", seconds=secondsF2,
            note="kk kh -f Z2 (native, no UCT); mirrored to canonical",
            agree=_agreeGroups(knot, "khovanov_homology", f2)),
    }


# ---- SnapPy (hyperbolic volume) ------------------------------------------------------------

def snappyRun(knot, reps):
    """SnapPy's hyperbolic volume and Chern-Simons invariant for a hyperbolic knot (by its
    KnotInfo name), each from its own timed call. Neither is a native invariant, so both are
    oracle-only data (no agreement verdict). Non-hyperbolic knots (torus knots, etc.) report n/a
    rather than a number read off a degenerate solution: SnapPy will return a Chern-Simons value
    for a torus knot from a triangulation with flat tetrahedra, and that is not published here."""
    import snappy
    name = getattr(knot, "identity", None)
    names = ("hyperbolic_volume", "chern_simons")
    if not name:
        miss = Measurement(value="n/a", seconds=None, note="no KnotInfo name for SnapPy",
                           agree="n/a")
        return {row: miss for row in names}

    def geometric(read):
        def call():
            manifold = snappy.Manifold(str(name))
            solution = manifold.solution_type()
            if solution != "all tetrahedra positively oriented":
                raise ValueError("non-geometric solution: %s" % solution)
            return float(read(manifold))
        return call

    rows = {}
    for row, read, label in (("hyperbolic_volume", lambda m: m.volume(), "volume()"),
                             ("chern_simons", lambda m: m.chern_simons(), "chern_simons()")):
        try:
            value, seconds = _best(geometric(read), reps)
        except Exception as error:
            rows[row] = Measurement(value="n/a", seconds=None,
                                    note="not hyperbolic (%s)" % type(error).__name__,
                                    agree="n/a")
        else:
            # round-then-add-zero turns a -1e-17 residue (an amphichiral knot's CS) into 0.0
            rows[row] = Measurement(value=f"{round(value, 10) + 0.0:.10f}", seconds=seconds,
                                    note=f"snappy Manifold(name).{label}", agree="oracle")
    return rows


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


def _determinantFromAlexander(alexander):
    """|Alexander(-1)| for an Alexander polynomial given as {exponent: coefficient}."""
    return abs(sum(coefficient * (-1) ** exponent for exponent, coefficient in alexander.items()))


def reginaRun(knot, reps):
    """regina's Jones (a Laurent poly in x = t^(1/2)) mapped to native's t by halving exponents,
    judged up to the t <-> t^-1 convention (regina matches native directly). HOMFLY is reported as
    oracle-only data -- native computes no HOMFLY. regina reads the PD via Link.fromPD.

    Alexander comes from its own timed call, so the Jones cell's time is not inflated by it. It
    is judged up to the +/- t^k unit, and the determinant is |Alexander(-1)| from the same call --
    the same derivation the regina validator uses. Both matched native on 3_1, 4_1, 5_2, 8_19
    and 10_124. Regina has no signature."""
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
    rows = {
        "jones_polynomial": Measurement(
            value=_shortValue(jones), seconds=seconds, note="regina jones(); x = t^1/2",
            agree=_agreeJones(knot, jones)),
        "homfly_polynomial": Measurement(
            value=homflyText.strip()[:40] or "?", seconds=None,
            note="regina homfly(); oracle-only", agree="oracle"),
    }
    try:
        alexanderText, secondsAlexander = _best(
            lambda: str(regina.Link.fromPD(pd).alexander()), reps)
        alexander = _parseLaurent(alexanderText, "x")
        if not alexander:
            raise ValueError("unexpected regina Alexander: %r" % alexanderText)
        determinant = _determinantFromAlexander(alexander)
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        rows.update({"alexander_polynomial": miss, "determinant": miss})
    else:
        rows.update({
            "alexander_polynomial": Measurement(
                value=_shortValue(alexander), seconds=secondsAlexander,
                note="regina alexander()", agree=_agreeAlexander(knot, alexander)),
            "determinant": Measurement(
                value=str(determinant), seconds=None,
                note="same regina alexander() call; |Alexander(-1)|",
                agree=_agreeScalar(knot, "determinant", determinant, mirror=lambda v: v)),
        })
    return rows


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


def _parseSageFields(text, tags=None):
    """Pull the tagged structured lines a sage script prints into {tag: value}."""
    fields = {}
    for line in text.splitlines():
        for tag in (_SAGE_TAGS if tags is None else tags):
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


def _sageNativeRun(knot, reps):
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


# ---- SageMath target rows (no native engine) ------------------------------------------------
#
# Probed on CT 250 (scripts/probe_sage_targets.py, SageMath 9.5): homfly_polynomial,
# omega_signature and Knot.arf_invariant exist; conway_polynomial, kauffman_polynomial,
# signature_function, algebraic_concordance_order and braid_index do not. These rows get their
# own sage run so the native-row cells' times are not inflated, and each operation is timed
# INSIDE sage (best of reps), because a whole-subprocess time here would be almost entirely the
# runtime boot rather than the computation a future engine has to beat.

_SAGE_TARGET_SCRIPT = """import time
L = Link(%(pd)s)
K = Knot(%(pd)s)
def _timed(f):
    best = None
    for _ in range(%(reps)d):
        start = time.perf_counter()
        value = f()
        elapsed = time.perf_counter() - start
        best = elapsed if best is None or elapsed < best else best
    return value, float(best)
H, tH = _timed(lambda: L.homfly_polynomial())
print("HOMFLY", repr(str(H)))
print("HOMFLY_SECONDS", tH)
W, tW = _timed(lambda: [int(L.omega_signature(exp(2*pi*I*k/12))) for k in range(1, 7)])
print("OMEGA_SIGNATURE", W)
print("OMEGA_SECONDS", tW)
print("SIGNATURE", int(L.signature()))
A, tA = _timed(lambda: int(K.arf_invariant()))
print("ARF", A)
print("ARF_SECONDS", tA)
"""
_SAGE_TARGET_TAGS = ("HOMFLY", "HOMFLY_SECONDS", "OMEGA_SIGNATURE", "OMEGA_SECONDS", "SIGNATURE",
                     "ARF", "ARF_SECONDS")
_SAGE_ABSENT = ("conway_polynomial", "kauffman_polynomial", "algebraic_concordance_order",
                "braid_index")


def _canonicalSignatureFunction(samples, signature):
    """Sage's Levine-Tristram samples at e^(2 pi i k/12), k = 1..6, in the canonical sign.

    Sage's signature is the negation of native's (verified by scripts/verify_sage_conventions.py),
    and omega_signature is the same form evaluated at omega, so the same negation applies -- but
    only after checking the sample at omega = -1 (k = 6) IS sage's own signature(). If it is not,
    the samples are not what this assumes and it raises."""
    if len(samples) != 6:
        raise ValueError(f"expected 6 Levine-Tristram samples, got {samples!r}")
    if samples[-1] != signature:
        raise ValueError(f"omega_signature(-1) = {samples[-1]} but signature() = {signature}")
    return [-value for value in samples]


def _sageTargetRun(knot, reps):
    """One sage run for the Sage-group target rows: HOMFLY-PT, the Levine-Tristram signature
    function (sampled at e^(2 pi i k/12), k = 1..6, canonical sign), and Arf. All oracle-only.
    The rows SageMath 9.5 lacks report n/a with that reason rather than 'adapter pending'."""
    import os
    import tempfile
    rows = {name: Measurement(value="n/a", seconds=None,
                              note="not in SageMath 9.5 (probed); no provisioned oracle has it",
                              agree="n/a")
            for name in _SAGE_ABSENT}
    try:
        script = _SAGE_TARGET_SCRIPT % {"pd": repr(pdAsList(knot)), "reps": max(reps, 1)}
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "targets.sage")
            with open(path, "w") as handle:
                handle.write(script)
            out = subprocess.run(["sage", path], check=True, capture_output=True, text=True).stdout
        fields = _parseSageFields(out, _SAGE_TARGET_TAGS)
        missing = [tag for tag in _SAGE_TARGET_TAGS if tag not in fields]
        if missing:
            raise ValueError("sage target output missing %s" % ", ".join(missing))
        levineTristram = _canonicalSignatureFunction(fields["OMEGA_SIGNATURE"], fields["SIGNATURE"])
    except Exception as error:
        miss = Measurement(value=f"error: {type(error).__name__}", seconds=None,
                           note=str(error)[:80], agree="n/a")
        rows.update({name: miss for name in
                     ("homfly_polynomial", "signature_function", "arf_invariant")})
        return rows
    inside = "timed inside sage, runtime boot excluded"
    rows.update({
        "homfly_polynomial": Measurement(
            value=fields["HOMFLY"][:40], seconds=fields["HOMFLY_SECONDS"],
            note=f"sage homfly_polynomial() in (L, M); {inside}", agree="oracle"),
        "signature_function": Measurement(
            value=f"k=1..6: {levineTristram}", seconds=fields["OMEGA_SECONDS"],
            note=f"sage omega_signature at e^(2 pi i k/12), negated to canonical; {inside}",
            agree="oracle"),
        "arf_invariant": Measurement(
            value=str(fields["ARF"]), seconds=fields["ARF_SECONDS"],
            note=f"sage Knot.arf_invariant(); {inside}", agree="oracle"),
    })
    return rows


def sageRun(knot, reps):
    """Both sage runs: the native rows (one run, whole-subprocess time) and the target rows
    (a second run, each operation timed inside sage)."""
    rows = _sageNativeRun(knot, reps)
    rows.update(_sageTargetRun(knot, reps))
    return rows


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


def knotkitAvailable():
    return _probeBinary("kk", "knotkit")


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


def knotkitVersion():
    return _gitShaVersion("knotkit")


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
    Oracle("khoca", khocaAvailable, khocaRun, khocaVersion),
    Oracle("knotkit", knotkitAvailable, knotkitRun, knotkitVersion),
]
