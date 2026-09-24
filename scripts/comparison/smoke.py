#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Run one roster oracle's measurement directly on named knots and print every row.

Artifact generation is the slow, all-oracle path; this is the quick check that a single
oracle's run function works on a host before spending a generation on it -- the only way
to exercise an oracle provisioned on CT 250 but not in the sandbox (Sage):

    git pull ; python tools/ct_exec.py -- "cd /opt/tetradrome/src && \\
        git fetch --depth 1 origin claude && git reset --hard FETCH_HEAD && \\
        /opt/tetradrome/venv/bin/python scripts/comparison/smoke.py sage 3_1 4_1 8_19"

Exit codes compose with ct_exec: 0 = every row produced a value (n/a with a stated reason
counts), 1 = at least one row reported an error, 2 = unknown or unavailable oracle.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

import adapters  # noqa: E402
from tetradrome import knots  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: smoke.py <oracle key> <knot name> [<knot name> ...]", file=sys.stderr)
        return 2
    byKey = {oracle.key: oracle for oracle in adapters.ORACLES}
    oracle = byKey.get(argv[0])
    if oracle is None or oracle.run is None:
        print(f"unknown oracle or no run function: {argv[0]} (have {sorted(byKey)})",
              file=sys.stderr)
        return 2
    available, detail = oracle.available()
    print(f"{oracle.key}: {detail}; version {oracle.version()}")
    if not available:
        return 2
    failed = False
    for name in argv[1:]:
        print(f"=== {name}")
        for row, measurement in oracle.run(knots.from_name(name), 1).items():
            seconds = "-" if measurement.seconds is None else f"{measurement.seconds * 1000:.1f}ms"
            print(f"  {row:30} {measurement.agree:8} {seconds:>10}  {measurement.value}"
                  f"  [{measurement.note}]")
            failed = failed or measurement.value.startswith("error:")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
