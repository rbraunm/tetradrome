# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""knotkit as a SPEC 12.1 validator: the Rasmussen s-invariant.

Self-contained by design: the comparison layer (``scripts/comparison/adapters.py``)
keeps its own measurement-oriented knotkit path, and the two deliberately do not share
code -- the operator's standing call. Do not unify them.

knotkit is a C++ binary (``kk``, provisioned from source by
``scripts/install_oracles.sh``). One ``kk s -f Q <PD>`` subprocess reads the knot's PD
bracket string verbatim and prints a single line ``s(<input>; Q) = n``.

Field: Q, deliberately, not kk's Z2 default. Native ``rasmussen_s`` is Rasmussen's
original invariant, read off Lee homology over Q, so the checking value must be the same
object. (The two fields agreed on every knot in the chiral sweep, but they are not equal
in general.)

Convention, verified empirically on the chiral sweep (3_1, 4_1, 5_2, 8_19, 10_124) in both
fields before wiring: fed our PD, knotkit's s is the NEGATION of canonical, so the fixed
transform is s -> -s -- the same convention as KnotJob. Negation matched 10/10 cells;
direct matched only the amphichiral 4_1. Probes must feed ``knot.pd_code``: a PD literal
copied from elsewhere can be the mirror of ours and silently invert every sign (see
roadmap/research/knotkit.md).

knotkit also computes Khovanov homology over Z2 and Q. That is measured in the benchmark
but deliberately not wired here: those invariants already carry three independent
computed validators, and each added validator runs inside every strict compute call.

A broken run raises; None is reserved for inputs knotkit cannot check at all (uncovered
invariant, or a knot with no PD).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

_COVERED = {"rasmussen_s"}
_S_LINE = re.compile(r"^s\(.*\) = (-?\d+)$")


def _oracle_home() -> str:
    return os.environ.get("ORACLE_HOME", "/opt/oracles")


def _bracket(knot) -> str:
    return "PD[" + ",".join("X[%d,%d,%d,%d]" % tuple(c) for c in knot.pd_code) + "]"


def raw_knotkit_s(knot) -> str:
    """One ``kk s -f Q`` run on the knot's PD; returns its stdout."""
    return subprocess.run(
        ["kk", "s", "-f", "Q", _bracket(knot)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def parse_s(text: str) -> int:
    """``kk s`` stdout -> s in knotkit's RAW convention. Exactly one ``s(...) = n``
    line, or it raises."""
    matches = (_S_LINE.match(line.strip()) for line in text.splitlines())
    values = [int(match.group(1)) for match in matches if match]
    if len(values) != 1:
        raise ValueError(f"expected one s line in kk output, found {len(values)}: {text[:80]!r}")
    return values[0]


class KnotkitValidator:
    """Read-only cross-check against knotkit (SPEC 12.1, ADR 0006)."""

    name = "knotkit"
    covered_invariants = _COVERED

    def is_available(self) -> bool:
        return shutil.which("kk") is not None

    def version_info(self) -> dict:
        """The built source's git sha, mirroring the install_oracles.sh derivation."""
        done = subprocess.run(
            ["git", "-C", os.path.join(_oracle_home(), "knotkit"), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        sha = done.stdout.strip()
        return {"knotkit": f"git:{sha}" if done.returncode == 0 and sha else "absent"}

    def known_value(self, knot, invariant: str):
        """knotkit's value under the canonical name and convention, or None when knotkit
        cannot check this input (uncovered invariant, or a knot with no PD)."""
        if invariant not in _COVERED or not knot.pd_code:
            return None
        return -parse_s(raw_knotkit_s(knot))
