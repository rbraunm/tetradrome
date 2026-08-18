# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Sage as a SPEC 12.1 validator: the classical four plus Khovanov Q/F2 cross-checks.

Sage is provisioned only on CT 250 (apt sagemath, INSTALL_SAGE=1), so on other hosts
this validator is wired-but-not-installed: strict raises the provisioning gap for any
invariant no other installed oracle covers (signature is the one such invariant), and
the artifact preflight refuses to generate. That is deliberate -- absence is loud,
never silently skipped.

CONVENTIONS ARE VERIFIED, not assumed. ``scripts/verify_sage_conventions.py`` -- which
imports THIS module's sage invocation and transforms, so what it verifies is literally
the code wired here -- returned CONSISTENT on CT 250 (SageMath version 9.5) on
2026-08-17 over the chiral+amphichiral sweep 3_1/4_1/5_2/8_19/10_124 (Khovanov fields:
minus 10_124, untested in sage there). The verdicts below are that run's transcription;
re-run the probe after any sage upgrade before trusting a changed environment.

Self-contained with respect to the comparison layer by design (the operator's call,
same as regina_adapter and knotjob_adapter -- do not unify with sageRun). Cross-package
imports are deferred to call time (the registry may import this module while
``invariants`` is mid-initialization).
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import tempfile

_COVERED = {
    "signature", "determinant", "alexander_polynomial", "jones_polynomial",
    "rational_khovanov_homology", "khovanov_homology",
}

# The probe's verdict per invariant, transcribed verbatim from the CONSISTENT
# CT 250 run of scripts/verify_sage_conventions.py (SageMath version 9.5, 2026-08-17;
# signature/jones read as mirror, everything else direct up to canonicalization).
_VERIFIED_VERDICTS: dict[str, str] = {
    "signature": "NEGATED",
    "determinant": "DIRECT",
    "alexander_polynomial": "CANONICAL",
    "jones_polynomial": "NEGATED_EXPONENTS",
    "rational_khovanov_homology": "DIRECT",
    "khovanov_homology": "DIRECT",
}

# ---- sage invocation (single source: the probe imports these) -----------------------

# One tagged print line per field, composed per request so a signature check never
# pays for sage's (slow) Khovanov computation. The KHOVANOV cells are
# {(h, q): invariant-factor tuple}, 0 = a Z summand.
_TAG_LINES = {
    "JONES": (
        "J = L.jones_polynomial()\n"
        'print("JONES", {int(e): int(c) for c, e in J.coefficients()})'
    ),
    "ALEXANDER": (
        "A = L.alexander_polynomial()\n"
        'print("ALEXANDER", {int(k): int(v) for k, v in A.dict().items()})'
    ),
    "SIGNATURE": 'print("SIGNATURE", int(L.signature()))',
    "DETERMINANT": 'print("DETERMINANT", int(L.determinant()))',
    "KHOVANOV": (
        "K = L.khovanov_homology()\n"
        "kh = {}\n"
        "for q in K:\n"
        "    for h in K[q]:\n"
        "        inv = tuple(int(x) for x in K[q][h].invariants())\n"
        "        if inv:\n"
        "            kh[(int(h), int(q))] = inv\n"
        'print("KHOVANOV", kh)'
    ),
}

_NEEDED_TAGS = {
    "signature": ("SIGNATURE",),
    "determinant": ("DETERMINANT",),
    "alexander_polynomial": ("ALEXANDER",),
    "jones_polynomial": ("JONES",),
    "rational_khovanov_homology": ("KHOVANOV",),
    "khovanov_homology": ("KHOVANOV",),
}


def sage_fields(knot, tags) -> dict:
    """One sage run on the knot's PD computing exactly ``tags`` -> {tag: parsed value}.
    Fails loud on a nonzero exit or any missing tag."""
    pd = repr([list(crossing) for crossing in knot.pd_code])
    script = "L = Link(%s)\n%s\n" % (pd, "\n".join(_TAG_LINES[tag] for tag in tags))
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "compute.sage")
        with open(path, "w") as handle:
            handle.write(script)
        proc = subprocess.run(["sage", path], check=True, capture_output=True, text=True)
    fields = {}
    for line in proc.stdout.splitlines():
        for tag in tags:
            if line.startswith(tag + " "):
                fields[tag] = ast.literal_eval(line[len(tag) + 1:])
    missing = [tag for tag in tags if tag not in fields]
    if missing:
        raise ValueError("sage output missing %s" % ", ".join(missing))
    return fields


# ---- canonicalization helpers -------------------------------------------------------

def _canonical_jones(poly: dict[int, int]):
    from ..invariants import jones

    low, high = min(poly), max(poly)
    return jones.canonical_laurent(low, [poly.get(e, 0) for e in range(low, high + 1)])


def _canonical_alexander(poly: dict[int, int]):
    from ..invariants import seifert

    low, high = min(poly), max(poly)
    return seifert.canonical_alexander([poly.get(e, 0) for e in range(low, high + 1)])


def _free_and_torsion(cells):
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


def _f2_from_integral(free, torsion):
    f2 = dict(free)
    for (h, q), count in torsion.items():
        f2[(h, q)] = f2.get((h, q), 0) + count
        f2[(h - 1, q)] = f2.get((h - 1, q), 0) + count
    return {key: rank for key, rank in f2.items() if rank}


def _mirror_groups(groups):
    return {(-h, -q): rank for (h, q), rank in groups.items()}


# ---- candidate transforms (single source: the probe derives its candidates here) ----

# Keyed (invariant, verdict label): every label the probe can print for an invariant
# maps to the transform that label means. Verification picks exactly one label per
# invariant; wiring records it in _VERIFIED_VERDICTS.
_TRANSFORMS = {
    ("signature", "DIRECT"): lambda f: f["SIGNATURE"],
    ("signature", "NEGATED"): lambda f: -f["SIGNATURE"],
    ("determinant", "DIRECT"): lambda f: f["DETERMINANT"],
    ("alexander_polynomial", "CANONICAL"): lambda f: _canonical_alexander(f["ALEXANDER"]),
    ("jones_polynomial", "DIRECT"): lambda f: _canonical_jones(f["JONES"]),
    ("jones_polynomial", "NEGATED_EXPONENTS"): lambda f: _canonical_jones(
        {-e: c for e, c in f["JONES"].items()}
    ),
    ("rational_khovanov_homology", "DIRECT"): lambda f: _free_and_torsion(f["KHOVANOV"])[0],
    ("rational_khovanov_homology", "MIRRORED"): lambda f: _mirror_groups(
        _free_and_torsion(f["KHOVANOV"])[0]
    ),
    ("khovanov_homology", "DIRECT"): lambda f: _f2_from_integral(
        *_free_and_torsion(f["KHOVANOV"])
    ),
    ("khovanov_homology", "MIRRORED"): lambda f: _mirror_groups(
        _f2_from_integral(*_free_and_torsion(f["KHOVANOV"]))
    ),
}


def candidate_labels(invariant: str) -> tuple[str, ...]:
    """The verdict labels the probe tests for ``invariant``, from the transform table."""
    return tuple(label for (name, label) in _TRANSFORMS if name == invariant)


class SageValidator:
    """Read-only cross-check against Sage (SPEC 12.1, ADR 0006), transforms verified
    on CT 250 (see the module docstring for the run's provenance)."""

    name = "sage"
    covered_invariants = _COVERED

    def is_available(self) -> bool:
        return shutil.which("sage") is not None

    def version_info(self) -> dict:
        """First line of ``sage --version`` up to the comma, mirroring the
        install_oracles.sh derivation; 'absent' when sage is not on PATH."""
        if not self.is_available():
            return {"sage": "absent"}
        proc = subprocess.run(["sage", "--version"], capture_output=True, text=True)
        line = (proc.stdout or proc.stderr).strip().splitlines()[0]
        return {"sage": line.split(",", 1)[0].strip()}

    def known_value(self, knot, invariant: str):
        """Sage's value under the verified convention, or None when sage cannot check
        this input (uncovered invariant, or a knot with no PD)."""
        if invariant not in _COVERED or not knot.pd_code:
            return None
        label = _VERIFIED_VERDICTS[invariant]
        fields = sage_fields(knot, _NEEDED_TAGS[invariant])
        return _TRANSFORMS[(invariant, label)](fields)
