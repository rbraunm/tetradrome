#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Probe which of the benchmark's Sage target rows this Sage can actually compute.

Run this ON CT 250 (the only provisioned host with sage) under the plain venv python:

    python tools/ct_exec.py -- "cd /opt/tetradrome/src && \\
        git fetch --depth 1 origin claude && git reset --hard FETCH_HEAD && \\
        /opt/tetradrome/venv/bin/python scripts/probe_sage_targets.py"

The target rows (Conway, HOMFLY-PT, Kauffman, Levine-Tristram signature function, Arf,
algebraic concordance order, braid index) have no native engine, so there is no
convention to verify against -- this only establishes, per candidate, whether the method
exists in this Sage, what it returns, and how long it takes, so the comparison layer's
sageRun can be extended with exactly the ones that work. Nothing is wired from a guess.

It also prints Sage's full public method list for Link and Knot, so a capability under a
name not guessed below still shows up in the one round trip.

Output is plain text lines meant to be pasted back verbatim. Exit codes compose with
ct_exec: 0 = sage ran (individual candidates may still have failed -- that is data),
2 = sage missing or the probe script itself broke.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tetradrome import knots  # noqa: E402

SWEEP = ["3_1", "4_1", "5_2", "8_19", "10_124"]

# (label, sage expression over L = Link(pd) and K = Knot(pd)). Each is tried independently.
CANDIDATES = [
    ("conway_polynomial", "L.conway_polynomial()"),
    ("homfly_polynomial", "L.homfly_polynomial()"),
    ("kauffman_polynomial", "L.kauffman_polynomial()"),
    ("omega_signature", "[L.omega_signature(exp(2*pi*I*k/12)) for k in range(1, 6)]"),
    ("signature_function", "L.signature_function()"),
    ("arf_invariant", "K.arf_invariant()"),
    ("algebraic_concordance_order", "K.algebraic_concordance_order()"),
    ("braid_index", "L.braid_index()"),
    ("seifert_matrix", "L.seifert_matrix()"),
]

_SCRIPT_HEAD = '''import time
L = Link(%(pd)s)
K = Knot(%(pd)s)
print("SAGE_VERSION", version())
'''

_SCRIPT_CANDIDATE = '''try:
    _t = time.perf_counter()
    _v = %(expression)s
    print("CANDIDATE", %(label)r, "ok", "%%.4fs" %% (time.perf_counter() - _t), repr(_v)[:200])
except Exception as _e:
    print("CANDIDATE", %(label)r, "error", type(_e).__name__, str(_e)[:160].replace("\\n", " "))
'''

_SCRIPT_METHODS = '''print("METHODS_LINK", sorted(m for m in dir(L) if not m.startswith("_")))
print("METHODS_KNOT", sorted(m for m in dir(K) if not m.startswith("_")))
'''


def sage_script(pd, with_methods: bool) -> str:
    parts = [_SCRIPT_HEAD % {"pd": pd}]
    for label, expression in CANDIDATES:
        parts.append(_SCRIPT_CANDIDATE % {"label": label, "expression": expression})
    if with_methods:
        parts.append(_SCRIPT_METHODS)
    return "".join(parts)


def main() -> int:
    if shutil.which("sage") is None:
        print("sage not on PATH -- run this on CT 250", file=sys.stderr)
        return 2
    for index, name in enumerate(SWEEP):
        pd = [list(crossing) for crossing in knots.from_name(name).pd_code]
        print(f"=== {name}", flush=True)
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "probe.sage")
            with open(path, "w") as handle:
                handle.write(sage_script(pd, with_methods=(index == 0)))
            done = subprocess.run(["sage", path], capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout)
            print(done.stderr[-2000:], file=sys.stderr)
            print(f"probe script failed on {name} (exit {done.returncode})", file=sys.stderr)
            return 2
        print(done.stdout, end="", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
