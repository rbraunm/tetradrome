# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Exact reduction by Gaussian elimination of the complex (engine Phase 4).

The Gaussian elimination lemma: if the differential has a unit entry a -> b (a a
generator of C^n, b of C^(n+1)), the complex is chain homotopy equivalent to the one
with a and b deleted, where every surviving column x picks up the zig-zag correction

    d'(x) = d(x) - <d(x), b> * lambda^{-1} * d(a)     (then drop the a, b components),

lambda = <d(a), b>. This is exact -- identical homology and gradings -- and assumes
nothing about the knot (decision 0007); delooping is already implicit in our generators
(each resolution circle contributes the two V-labelled basis elements). Over a field,
cancelling every unit leaves the zero differential, so the surviving generators ARE the
homology: an independent homology algorithm that must agree with the rank-counting
reference (`raw == reduced`), and the basis for shrinking a complex toward its homology
size (the engine's memory tool, design section 5).

Field-agnostic: p = 2 for F2 (the only unit is 1), p = None for exact Q (Fraction).
Slower than rank counting and unoptimized (it rescans for a pivot each step); it is a
reference cross-check, not the fast path -- packed/accelerated reduction is Phase 5.
"""
from __future__ import annotations

from fractions import Fraction

from .complex import GradedComplex
from .rational_complex import RationalComplex


def _inv(v, p):
    if p is None:
        return Fraction(1) / v
    return pow(int(v) % p, p - 2, p)        # Fermat inverse in F_p (p prime)


def _norm(v, p):
    return v if p is None else int(v) % p


def _cancel_all(dims, columns, p):
    """Reduce to homology by cancelling every unit. `columns[n]` is the list of d^n
    columns as {row: coeff} maps; returns {degree: dim H^n}. Generators are tracked by
    id = (degree, position) so deleting one needs no reindexing.
    """
    cols: dict[int, dict[tuple, dict[tuple, object]]] = {}
    for n, d in dims.items():
        cols[n] = {}
        col_list = columns.get(n, [])
        for pos in range(d):
            raw = col_list[pos] if pos < len(col_list) else {}
            cols[n][(n, pos)] = {
                (n + 1, r): _norm(c, p) for r, c in raw.items() if _norm(c, p) != 0
            }

    while True:
        pivot = None
        for n, gens in cols.items():
            for a_id, col in gens.items():
                if col:
                    pivot = (n, a_id, next(iter(col)))
                    break
            if pivot:
                break
        if pivot is None:
            break

        n, a_id, b_id = pivot
        a_col = cols[n][a_id]
        lam_inv = _inv(a_col[b_id], p)
        for x_id, x_col in cols[n].items():
            if x_id == a_id:
                continue
            coef = x_col.get(b_id)
            if not coef:
                continue
            factor = coef * lam_inv
            for t_id, c in a_col.items():
                nv = _norm(x_col.get(t_id, 0) - factor * c, p)
                if nv == 0:
                    x_col.pop(t_id, None)
                else:
                    x_col[t_id] = nv

        del cols[n][a_id]                       # remove a from C^n
        if n + 1 in cols:
            cols[n + 1].pop(b_id, None)         # remove b from C^(n+1) (out-of-b dropped)
        if n - 1 in cols:
            for x_col in cols[n - 1].values():  # into-a dropped
                x_col.pop(a_id, None)
        for x_col in cols[n].values():          # clear any residual b component
            x_col.pop(b_id, None)

    return {n: len(gens) for n, gens in cols.items() if gens}


def gaussian_homology(cx) -> dict[int, int]:
    """Homology of `cx` by exact Gaussian cancellation (an independent check on the
    rank-based reducers). Accepts an F2 GradedComplex or a RationalComplex."""
    if not isinstance(cx, (GradedComplex, RationalComplex)):
        raise TypeError(f"gaussian_homology: unsupported complex type {type(cx).__name__}.")
    dims = {n: cx.dim(n) for n in cx.degrees()}
    if isinstance(cx, GradedComplex):
        columns = {n: [dict.fromkeys(col, 1) for col in cx.differential(n)] for n in dims}
        return _cancel_all(dims, columns, 2)
    columns = {n: [dict(col) for col in cx.differential(n)] for n in dims}
    return _cancel_all(dims, columns, None)
