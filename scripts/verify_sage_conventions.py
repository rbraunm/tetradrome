#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Verify Sage's invariant conventions against native, per knot, before wiring.

Run this ON CT 250 (the only provisioned host with sage) under the plain venv python:

    python tools/ct_exec.py -- "cd /opt/tetradrome/src && \\
        git fetch --depth 1 origin claude && git reset --hard FETCH_HEAD && \\
        /opt/tetradrome/venv/bin/python scripts/verify_sage_conventions.py"

This probe is the fail-first gate in front of ``backends/sage_adapter.SageValidator``.
It imports the validator's OWN sage invocation and candidate transforms, so what is
verified here is literally the code that gets wired -- the only thing transcribed at
wiring time is the verdict label per invariant (into ``_VERIFIED_VERDICTS``). For each
covered invariant it compares sage's value under every candidate transform against the
native one, over a chiral + amphichiral sweep (a stray sign flip or t <-> t^-1 cannot
pass). Expected, per the comparison layer's measured knowledge: signature NEGATED,
Jones NEGATED_EXPONENTS, Alexander CANONICAL, determinant DIRECT, Khovanov Q/F2 DIRECT.

Exit codes compose with ct_exec: 0 = every invariant consistent across its sweep (safe
to transcribe the printed verdicts), 1 = mixed or ambiguous (do NOT wire), 2 = sage
missing or broken.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tetradrome import invariants, knots  # noqa: E402
from tetradrome.backends import sage_adapter  # noqa: E402

CLASSICAL_INVARIANTS = [
    "signature", "determinant", "alexander_polynomial", "jones_polynomial",
]
KHOVANOV_INVARIANTS = ["rational_khovanov_homology", "khovanov_homology"]
CLASSICAL_SWEEP = ["3_1", "4_1", "5_2", "8_19", "10_124"]
KHOVANOV_SWEEP = ["3_1", "4_1", "5_2", "8_19"]  # sage khovanov proven on the ladder;
                                                # 10_124 is untested there, so skipped.


def collect_fields():
    """One sage run per knot, computing exactly the tags its sweeps need (10_124 skips
    KHOVANOV, sage's slow field). Fails loud on any broken run."""
    def tags_for(invariant_names):
        merged: list[str] = []
        for invariant in invariant_names:
            for tag in sage_adapter._NEEDED_TAGS[invariant]:
                if tag not in merged:
                    merged.append(tag)
        return tuple(merged)

    classical_tags = tags_for(CLASSICAL_INVARIANTS)
    all_tags = tags_for(CLASSICAL_INVARIANTS + KHOVANOV_INVARIANTS)
    fields_by_knot = {}
    for name in CLASSICAL_SWEEP:
        tags = all_tags if name in KHOVANOV_SWEEP else classical_tags
        print("  sage run: %s (%s)" % (name, ", ".join(tags)), flush=True)
        fields_by_knot[name] = sage_adapter.sage_fields(knots.from_name(name), tags)
    return fields_by_knot


def judge(invariant, sweep, fields_by_knot):
    """Per knot, print sage vs native under every candidate transform; return the
    verdict: the single transform label that held on every knot, or None."""
    labels = sage_adapter.candidate_labels(invariant)
    held = {label: True for label in labels}
    print("== %s ==" % invariant)
    for name in sweep:
        expected = invariants.compute(knots.from_name(name), invariant, validate="off").value
        outcomes = []
        for label in labels:
            value = sage_adapter._TRANSFORMS[(invariant, label)](fields_by_knot[name])
            ok = value == expected
            held[label] = held[label] and ok
            outcomes.append("%s: %s" % (label, "OK" if ok else "MISMATCH"))
        print("  %-7s native=%s | %s" % (name, _short(expected), "  ".join(outcomes)))
    winners = [label for label, ok in held.items() if ok]
    if len(winners) == 1:
        print("  verdict: %s (%d/%d knots)" % (winners[0], len(sweep), len(sweep)))
        return winners[0]
    if len(winners) > 1:
        # The sweep is chiral, so multiple surviving transforms is a probe defect;
        # refuse to bless either rather than guessing.
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

    fields_by_knot = collect_fields()
    verdicts = {}
    for invariant in CLASSICAL_INVARIANTS:
        verdicts[invariant] = judge(invariant, CLASSICAL_SWEEP, fields_by_knot)
    for invariant in KHOVANOV_INVARIANTS:
        verdicts[invariant] = judge(invariant, KHOVANOV_SWEEP, fields_by_knot)

    print("== VERDICTS ==")
    for invariant, verdict in verdicts.items():
        print("  %s: %s" % (invariant, verdict or "UNRESOLVED"))
    if all(verdicts.values()):
        print("RESULT: CONSISTENT -- transcribe these verdicts into "
              "sage_adapter._VERIFIED_VERDICTS to wire")
        return 0
    print("RESULT: UNRESOLVED -- do NOT wire; investigate the MIXED/AMBIGUOUS rows")
    return 1


if __name__ == "__main__":
    sys.exit(main())
