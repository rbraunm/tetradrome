# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Pure-Python Q reference reducer: homology of a RationalComplex by rank counting.

The rational counterpart of reduce_reference.f2_rank / homology, and the same faithful
reference standard (homology-engine design section 7) -- exact, obvious, slow; every
later accelerated reducer (multimodular, etc.) must agree with it. For a cochain complex
with d^n: C^n -> C^(n+1),

    dim H^n = (dim C^n - rank d^n) - rank d^(n-1),

so homology over Q reduces to rational ranks of the boundary maps.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from .rational_complex import RationalComplex


def rational_rank(columns: Sequence[Mapping[int, object]]) -> int:
    """Rank over Q of a rational matrix given column-wise as {row: coeff} maps.

    Column reduction over Q: each pivot is stored normalized to leading coefficient 1
    (keyed by its leading row, the maximum index). A new column is eliminated against
    existing pivots -- subtracting its leading coefficient times the matching pivot
    zeroes that row, and since a pivot's other entries lie strictly below its leading
    row, the column's leading row strictly drops each step, so the loop terminates. The
    column either reduces to zero (dependent) or becomes a new pivot.
    """
    pivots: dict[int, dict[int, Fraction]] = {}   # leading row -> normalized column
    rank = 0
    for col in columns:
        v = {int(r): Fraction(c) for r, c in col.items() if c != 0}
        while v:
            lead = max(v)
            piv = pivots.get(lead)
            if piv is None:
                inv = Fraction(1) / v[lead]
                pivots[lead] = {r: c * inv for r, c in v.items()}
                rank += 1
                break
            factor = v[lead]                       # piv[lead] == 1
            for r, c in piv.items():
                nv = v.get(r, Fraction(0)) - factor * c
                if nv == 0:
                    v.pop(r, None)
                else:
                    v[r] = nv
    return rank


def rational_homology(cx: RationalComplex, *, verify: bool = True) -> dict[int, int]:
    """Q homology of `cx`, as {degree: dim H^n} for every degree with non-zero homology.

    By default `cx` is checked for d^2 = 0 first (decision 0004); pass `verify=False` to
    skip that when already run. The negative-dimension backstop fires regardless.
    """
    if verify:
        cx.verify_d_squared()

    rank = {n: rational_rank(cx.differential(n)) for n in cx.degrees()}
    result: dict[int, int] = {}
    for n in cx.degrees():
        h = cx.dim(n) - rank.get(n, 0) - rank.get(n - 1, 0)
        if h < 0:
            raise RuntimeError(
                f"negative homology dimension {h} at degree {n}: not a valid complex "
                f"(d^2 != 0?)."
            )
        if h:
            result[n] = h
    return result
