# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Khoca as a SPEC 12.1 validator: unreduced Khovanov over F2 and over Q.

Self-contained by design: the comparison layer (``scripts/comparison/adapters.py``)
keeps its own measurement-oriented khoca path, and the two deliberately do not share
code -- the operator's standing call. Do not unify them.

Khoca is a pip module, not a subprocess: ``khoca.InteractiveCalculator(coefficient_ring=r)``
is called on the PD directly, with 0 = Z, 1 = Q, and a prime = F_p. Each field is
computed natively, so F2 comes straight from the ring-2 run rather than by the universal
coefficient theorem from an integral one -- unlike the KnotJob path. It is the fastest
homological oracle available here (0.02--0.09s in process), so there is deliberately NO
caching: every known_value call recomputes and the provenance record never claims a run
that did not happen.

Convention, verified empirically on the chiral sweep (3_1, 4_1, 5_2, 8_19, 10_124) across
both field rings before wiring: khoca's quantum grading is negated relative to KnotInfo's,
so the fixed transform to canonical is (h, q) -> (h, -q). q-negation matched 10/10 cells;
direct and full mirror each matched 0/10. Note this is a q-negation, NOT a mirror -- the
homological grading is untouched, so do not describe it as one.

The calculator returns ``[reduced, unreduced]``; only the unreduced half feeds the
canonical rows, since ``khovanov_homology`` and ``rational_khovanov_homology`` are both
unreduced theories (ADR 0001). The discarded reduced half is real, computed data and is
the cheapest available oracle for a future reduced row -- see homology-engine.md section
7, Phase 9, and roadmap/research/khtpp.md.

Rows are ``[t, q, torsionOrder, multiplicity]``. A field coefficient ring must never
produce a torsion row or a negative aggregate rank; both fail loud rather than being
filtered away. A broken run raises; None is reserved for inputs khoca cannot check at
all (uncovered invariant, or a knot with no PD).
"""
from __future__ import annotations

_COVERED = {"khovanov_homology", "rational_khovanov_homology"}

# Canonical invariant -> khoca coefficient_ring. F2 is the unmarked default per ADR 0001;
# each field is computed natively rather than derived from an integral run.
_RING_FOR_INVARIANT = {
    "khovanov_homology": 2,          # F_2
    "rational_khovanov_homology": 1,  # Q
}


def _unreduced_groups(out, ring: int) -> dict[tuple[int, int], int]:
    """The unreduced half of a khoca ``[reduced, unreduced]`` result -> {(h, q): dim}
    in the canonical q-convention."""
    groups: dict[tuple[int, int], int] = {}
    for t, q, torsion_order, multiplicity in out[1]:
        if torsion_order != 0:
            raise ValueError(
                f"field coefficient ring {ring} produced a torsion row "
                f"{(t, q, torsion_order, multiplicity)}"
            )
        key = (t, -q)
        groups[key] = groups.get(key, 0) + multiplicity
    groups = {key: rank for key, rank in groups.items() if rank}
    if any(rank < 0 for rank in groups.values()):
        raise ValueError(f"negative aggregate rank in khoca output: {groups}")
    if not groups:
        raise ValueError("empty Khovanov homology from khoca (never zero for a knot)")
    return groups


def raw_khoca(knot, ring: int) -> list:
    """One khoca call on the knot's PD; returns the raw ``[reduced, unreduced]`` pair.
    The reduced half is unused here -- see the module docstring."""
    import khoca

    pd = [list(crossing) for crossing in knot.pd_code]
    return khoca.InteractiveCalculator(coefficient_ring=ring)(pd)


class KhocaValidator:
    """Read-only cross-check against khoca (SPEC 12.1, ADR 0006)."""

    name = "khoca"
    covered_invariants = _COVERED

    def is_available(self) -> bool:
        try:
            import khoca  # noqa: F401
        except ImportError:
            return False
        return True

    def version_info(self) -> dict:
        """Pip distribution metadata, mirroring the install_oracles.sh derivation."""
        import importlib.metadata

        try:
            return {"khoca": importlib.metadata.version("khoca")}
        except importlib.metadata.PackageNotFoundError:
            return {"khoca": "absent"}

    def known_value(self, knot, invariant: str):
        """Khoca's value under the canonical name and convention, or None when khoca
        cannot check this input (uncovered invariant, or a knot with no PD)."""
        if invariant not in _COVERED or not knot.pd_code:
            return None
        ring = _RING_FOR_INVARIANT[invariant]
        return _unreduced_groups(raw_khoca(knot, ring), ring)
