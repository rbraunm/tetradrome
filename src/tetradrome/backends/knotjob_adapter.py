# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""KnotJob as a SPEC 12.1 validator: Khovanov over F2 and Q, and Rasmussen s.

Self-contained by design: the comparison layer (``scripts/comparison/adapters.py``)
keeps its own measurement-oriented KnotJob path, and the two deliberately do not share
code -- the operator's call, made when this validator was wired. Do not unify them.

One ``knotjob knot.txt -kb0 -s0`` subprocess yields everything at once: the integral
unreduced Khovanov homology (whose free part is the rational theory), the order-2
torsion (which gives the F2 theory via the universal coefficient theorem: a Z/2
summand at (h, q) contributes F2 classes at (h, q) and at (h-1, q)), and the
s-invariant. There is deliberately NO caching: a warm run costs ~0.2s (measured
across tier-0 through 10_124), so every known_value call invokes the jar and the
provenance record never claims a run that did not happen.

Convention, verified empirically on chiral and amphichiral knots (3_1, 4_1, 5_2,
8_19, 10_124) before wiring: KnotJob reads Tetradrome's PD in the mirror convention,
so the fixed transform to canonical is (h, q) -> (-h, -q) on the group dicts and
s -> -s. Any future disagreement raises through the validation machinery; a broken
run (missing binary output, unparseable text) raises rather than returning None --
None is reserved for inputs KnotJob cannot check at all (no PD).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

_COVERED = {"khovanov_homology", "rational_khovanov_homology", "rasmussen_s"}
_JAR_RELATIVE_PATH = "knotjob/KnotJob/KnotJob.jar"


def _oracle_home() -> str:
    return os.environ.get("ORACLE_HOME", "/opt/oracles")


def _bracket_pd(knot) -> str:
    """Tetradrome's PD as the ``PD[X[...],...]`` bracket string KnotJob reads."""
    return "PD[" + ",".join(
        "X[%d,%d,%d,%d]" % tuple(crossing) for crossing in knot.pd_code
    ) + "]"


def _field_after_colon(text: str, marker: str) -> str | None:
    """Text after ``:`` on the first line containing ``marker``, or None."""
    for line in text.splitlines():
        if marker in line:
            return line.split(":", 1)[1].strip()
    return None


def _monomial(term: str) -> tuple[tuple[int, int], int]:
    """One ``c t^h q^q`` monomial (factors ``*``-joined or juxtaposed, exponents
    optional and possibly negative) -> ((h, q), coefficient)."""
    term = term.replace("*", "").strip()
    leading = re.match(r"[+-]?\d+", term)
    coefficient = int(leading.group(0)) if leading else 1
    t_part = re.search(r"t(\^-?\d+)?", term)
    h = 0 if t_part is None else (int(t_part.group(1)[1:]) if t_part.group(1) else 1)
    q_part = re.search(r"q(\^-?\d+)?", term)
    q = 0 if q_part is None else (int(q_part.group(1)[1:]) if q_part.group(1) else 1)
    return (h, q), coefficient


def _parse_khovanov(text: str) -> dict[tuple[int, int], int]:
    """Sum of Khovanov monomials in t, q -> {(h, q): rank}, zeros dropped. An empty
    parse fails loud (unreduced Khovanov homology is never zero)."""
    cleaned = text.replace("(", "").replace(")", "").replace(" ", "")
    groups: dict[tuple[int, int], int] = {}
    for term in cleaned.split("+"):
        if not term:
            continue
        key, coefficient = _monomial(term)
        groups[key] = groups.get(key, 0) + coefficient
    groups = {key: rank for key, rank in groups.items() if rank}
    if not groups:
        raise ValueError(f"empty or unparseable Khovanov polynomial: {text!r}")
    return groups


def _f2_from_integral(free, torsion) -> dict[tuple[int, int], int]:
    """F2 Khovanov dimensions from the integral free ranks plus the order-2 torsion,
    by the universal coefficient theorem."""
    f2 = dict(free)
    for (h, q), count in torsion.items():
        f2[(h, q)] = f2.get((h, q), 0) + count
        f2[(h - 1, q)] = f2.get((h - 1, q), 0) + count
    return {key: rank for key, rank in f2.items() if rank}


def _mirror(groups) -> dict[tuple[int, int], int]:
    """Khovanov of the mirror knot: (h, q) -> (-h, -q)."""
    return {(-h, -q): rank for (h, q), rank in groups.items()}


def raw_knotjob(knot) -> str:
    """One ``knotjob -kb0 -s0`` run on the knot's PD; returns the output file's text.
    Fails loud on a nonzero exit or a missing output file."""
    bracket = _bracket_pd(knot)
    with tempfile.TemporaryDirectory() as work:
        with open(os.path.join(work, "knot.txt"), "w") as handle:
            handle.write(bracket + "\n")
        subprocess.run(
            ["knotjob", "knot.txt", "-kb0", "-s0"],
            cwd=work, check=True, capture_output=True, text=True,
        )
        out_path = os.path.join(work, "knot.txt_s0_kb0")
        if not os.path.exists(out_path):
            raise RuntimeError("knotjob wrote no knot.txt_s0_kb0 output file")
        with open(out_path) as handle:
            return handle.read()


class KnotJobValidator:
    """Read-only cross-check against KnotJob (SPEC 12.1, ADR 0006)."""

    name = "knotjob"
    covered_invariants = _COVERED

    def is_available(self) -> bool:
        return shutil.which("knotjob") is not None

    def version_info(self) -> dict:
        """Jar sha256, mirroring the install_oracles.sh / comparison-layer derivation."""
        import hashlib

        path = os.path.join(_oracle_home(), _JAR_RELATIVE_PATH)
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()[:12]
            return {"knotjob": f"sha256:{digest}"}
        except OSError:
            return {"knotjob": "absent"}

    def known_value(self, knot, invariant: str):
        """KnotJob's value under the canonical name and convention, or None when
        KnotJob cannot check this input (uncovered invariant, or a knot with no PD)."""
        if invariant not in _COVERED or not knot.pd_code:
            return None
        text = raw_knotjob(knot)
        if invariant == "rasmussen_s":
            s_text = _field_after_colon(text, "S-Invariant mod 0")
            if s_text is None:
                raise ValueError("no s-invariant line in knotjob output")
            return -int(s_text)
        free_text = _field_after_colon(text, "Integral unreduced Khovanov Homology")
        if free_text is None:
            raise ValueError("no integral Khovanov line in knotjob output")
        free = _parse_khovanov(free_text)
        if invariant == "rational_khovanov_homology":
            return _mirror(free)
        torsion_text = _field_after_colon(text, "Torsion of order 2")
        torsion = _parse_khovanov(torsion_text) if torsion_text else {}
        return _mirror(_f2_from_integral(free, torsion))
