#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Verify Sage's invariant conventions against native, per knot, before wiring.

Run this ON CT 250 (the only provisioned host with sage) under the plain venv python:

    python tools/ct_exec.py -- "cd /opt/tetradrome/src && \\
        git fetch --depth 1 origin claude && git reset --hard FETCH_HEAD && \\
        /opt/tetradrome/venv/bin/python scripts/verify_sage_conventions.py"

For each invariant a future SageValidator would cover, this compares sage's raw value
against the native one under every candidate transform, over a chiral + amphichiral
sweep (a stray sign flip or t <-> t^-1 cannot pass). The expected verdicts, pre-seeded
from the comparison layer's measured knowledge (sageRun docstring): signature NEGATED,
Jones NEGATED_EXPONENTS, Alexander CANONICAL, determinant DIRECT, Khovanov Q/F2 DIRECT.

The probe is deliberately self-contained -- the sage-side script and parsers are the
code shapes the SageValidator will carry, so what is verified here is what gets wired
(same doctrine as the Regina and KnotJob probes). Exit codes compose with ct_exec:
0 = every invariant consistent across the sweep (safe to wire the printed transforms),
1 = mixed or mismatched (do NOT wire), 2 = sage missing or broken.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tetradrome import invariants, knots  # noqa: E402
from tetradrome.invariants import jones as jones_mod  # noqa: E402
from tetradrome.invariants import seifert  # noqa: E402

CLASSICAL_SWEEP = ["3_1", "4_1", "5_2", "8_19", "10_124"]
KHOVANOV_SWEEP = ["3_1", "4_1", "5_2", "8_19"]  # sage khovanov proven on the ladder;
                                                # 10_124 is untested there, so skipped.

# Verbatim from the comparison layer's working sage path (KHOVANOV cells are
# {(h, q): invariant-factor tuple}, 0 = a Z summand).
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
_TAGS = ("JONES", "ALEXANDER", "SIGNATURE", "DETERMINANT", "KHOVANOV")


def sage_fields(knot) -> dict:
    """One sage run on the knot's PD -> the tagged structured fields. Fails loud."""
    script = _SAGE_SCRIPT % {"pd": repr([list(c) for c in knot.pd_code])}
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "compute.sage")
        with open(path, "w") as handle:
            handle.write(script)
        proc = subprocess.run(["sage", path], check=True, capture_output=True, text=True)
    fields = {}
    for line in proc.stdout.splitlines():
        for tag in _TAGS:
            if line.startswith(tag + " "):
                fields[tag] = ast.literal_eval(line[len(tag) + 1:])
    missing = [tag for tag in _TAGS if tag not in fields]
    if missing:
        raise ValueError("sage output missing %s" % ", ".join(missing))
    return fields


def free_and_torsion(cells):
    """Invariant factors -> (free ranks, order-2 torsion counts); even-order factors
    are the ones that survive to F2."""
    free, torsion = {}, {}
    for key, factors in cells.items():
        rank = sum(1 for x in factors if x == 0)
        even = sum(1 for x in factors if x != 0 and x % 2 == 0)
        if rank:
            free[key] = rank
        if even:
            torsion[key] = even
    return free, torsion


def f2_from_integral(free, torsion):
    f2 = dict(free)
    for (h, q), count in torsion.items():
        f2[(h, q)] = f2.get((h, q), 0) + count
        f2[(h - 1, q)] = f2.get((h - 1, q), 0) + count
    return {key: rank for key, rank in f2.items() if rank}


def mirror_groups(groups):
    return {(-h, -q): rank for (h, q), rank in groups.items()}


def canonical_jones(poly):
    low, high = min(poly), max(poly)
    return jones_mod.canonical_laurent(low, [poly.get(e, 0) for e in range(low, high + 1)])


def canonical_alexander(poly):
    low, high = min(poly), max(poly)
    return seifert.canonical_alexander([poly.get(e, 0) for e in range(low, high + 1)])


def native(knot, invariant):
    return invariants.compute(knot, invariant, validate="off").value


def judge(sweep, invariant, candidates):
    """Per knot, print sage vs native under every candidate transform; return the
    verdict: the single transform that held on every knot, or None."""
    held = {label: True for label, _ in candidates}
    print("== %s ==" % invariant)
    for name in sweep:
        knot = knots.from_name(name)
        fields = sage_fields(knot)
        expected = native(knot, invariant)
        outcomes = []
        for label, transform in candidates:
            value = transform(fields)
            ok = value == expected
            held[label] = held[label] and ok
            outcomes.append("%s: %s" % (label, "OK" if ok else "MISMATCH"))
        print("  %-7s native=%s | %s" % (name, _short(expected), "  ".join(outcomes)))
    winners = [label for label, ok in held.items() if ok]
    if len(winners) == 1:
        print("  verdict: %s (%d/%d knots)" % (winners[0], len(sweep), len(sweep)))
        return winners[0]
    if len(winners) > 1:
        # Amphichiral-only sweeps could be ambiguous; this sweep is chiral, so treat
        # multiple winners as a probe defect and refuse to bless either.
        print("  verdict: AMBIGUOUS (%s all held -- widen the sweep)" % ", ".join(winners))
        return None
    print("  verdict: MIXED -- no single transform held; do NOT wire")
    return None


def _short(value):
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def main() -> int:
    if shutil.which("sage") is None:
        print("sage is not on PATH -- run this on CT 250 (INSTALL_SAGE=1 host).")
        return 2
    banner = subprocess.run(["sage", "--version"], capture_output=True, text=True)
    print(banner.stdout.strip() or banner.stderr.strip())

    verdicts = {}
    verdicts["signature"] = judge(CLASSICAL_SWEEP, "signature", [
        ("DIRECT", lambda f: f["SIGNATURE"]),
        ("NEGATED", lambda f: -f["SIGNATURE"]),
    ])
    verdicts["determinant"] = judge(CLASSICAL_SWEEP, "determinant", [
        ("DIRECT", lambda f: f["DETERMINANT"]),
    ])
    verdicts["alexander_polynomial"] = judge(CLASSICAL_SWEEP, "alexander_polynomial", [
        ("CANONICAL", lambda f: canonical_alexander(f["ALEXANDER"])),
    ])
    verdicts["jones_polynomial"] = judge(CLASSICAL_SWEEP, "jones_polynomial", [
        ("DIRECT", lambda f: canonical_jones(f["JONES"])),
        ("NEGATED_EXPONENTS", lambda f: canonical_jones({-e: c for e, c in f["JONES"].items()})),
    ])
    verdicts["rational_khovanov_homology"] = judge(KHOVANOV_SWEEP, "rational_khovanov_homology", [
        ("DIRECT", lambda f: free_and_torsion(f["KHOVANOV"])[0]),
        ("MIRRORED", lambda f: mirror_groups(free_and_torsion(f["KHOVANOV"])[0])),
    ])
    verdicts["khovanov_homology"] = judge(KHOVANOV_SWEEP, "khovanov_homology", [
        ("DIRECT", lambda f: f2_from_integral(*free_and_torsion(f["KHOVANOV"]))),
        ("MIRRORED", lambda f: mirror_groups(f2_from_integral(*free_and_torsion(f["KHOVANOV"])))),
    ])

    print("== VERDICTS ==")
    for invariant, verdict in verdicts.items():
        print("  %s: %s" % (invariant, verdict or "UNRESOLVED"))
    if all(verdicts.values()):
        print("RESULT: CONSISTENT -- safe to wire the transforms above")
        return 0
    print("RESULT: UNRESOLVED -- do NOT wire; investigate the MIXED/AMBIGUOUS rows")
    return 1


if __name__ == "__main__":
    sys.exit(main())
