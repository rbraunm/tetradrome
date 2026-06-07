"""Pure-Python F2 reference reducer: homology of a GradedComplex by rank counting.

This is the faithful reference (homology-engine design section 7, Phase 1) --
correctness and obviousness over speed. Every later reducer (bit-packed F2, JIT, GPU)
must agree with it (SPEC.md 4.2 / 13.7). F2 only for now (decision 0003).

For a cochain complex with d^n: C^n -> C^(n+1),

    dim H^n = dim ker(d^n) - dim im(d^(n-1))
            = (dim C^n - rank d^n) - rank d^(n-1),

so homology reduces to F2 ranks of the boundary maps -- which is all this computes.
"""
from __future__ import annotations

from collections.abc import Sequence

from .complex import GradedComplex


def f2_rank(columns: Sequence[frozenset[int]]) -> int:
    """Rank over F2 of a 0/1 matrix given column-wise as sets of row indices.

    Column reduction over GF(2): each column is eliminated against the pivots found so
    far (keyed by their leading row); it either reduces to the zero vector (dependent,
    no new rank) or becomes a new pivot. The leading row strictly decreases on each
    elimination step, so the inner loop terminates.
    """
    pivots: dict[int, set[int]] = {}    # leading row -> reduced column
    rank = 0
    for col in columns:
        v = set(col)
        while v:
            lead = max(v)
            piv = pivots.get(lead)
            if piv is None:
                pivots[lead] = v        # new independent column
                rank += 1
                break
            v ^= piv                    # eliminate; leading row drops
    return rank


def homology(cx: GradedComplex, *, verify: bool = True) -> dict[int, int]:
    """F2 homology of the complex `cx`, as {degree: dim H^n} for every degree with
    non-zero homology (an omitted degree has H^n = 0).

    By default `cx` is checked for d^2 = 0 first (decision 0004): the homology of a
    non-complex is meaningless, so we refuse loudly rather than return a number. Pass
    `verify=False` to skip that O(edges) check when the caller has already run it --
    the impossible-result backstop below still fires regardless.
    """
    if verify:
        cx.verify_d_squared()

    rank = {n: f2_rank(cx.differential(n)) for n in cx.degrees()}
    result: dict[int, int] = {}
    for n in cx.degrees():
        h = cx.dim(n) - rank.get(n, 0) - rank.get(n - 1, 0)
        if h < 0:
            # rank(d^n) + rank(d^(n-1)) > dim C^n forces im d^(n-1) not contained in
            # ker d^n -- i.e. d^2 != 0. A Betti number is never negative; fail loud
            # even when the upfront check was skipped.
            raise RuntimeError(
                f"negative homology dimension {h} at degree {n}: not a valid complex "
                f"(d^2 != 0?)."
            )
        if h:
            result[n] = h
    return result
