# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Multimodular rational rank and homology (engine Phase 5).

The rational lane needs only homology *dimensions*, and for an integer matrix the rank over
ℚ equals the rank over F_p for every prime except the finitely many dividing a relevant
minor. A bad prime can only *lower* the rank (columns independent over ℚ may collapse mod
p, but never the reverse), so

    rank_ℚ(A) = max over several large primes p of rank_{F_p}(A mod p),

and a handful of large primes makes a wrong answer astronomically unlikely. No CRT and no
rational reconstruction are needed -- those reconstruct rational *entries*, which
dimension-only invariants (homology ranks, and Rasmussen s via filtered ranks) never use.

The point is speed: exact `Fraction` Gaussian elimination suffers coefficient explosion,
while mod-p arithmetic stays bounded. The result is validated identical to the exact
rational reducer across the catalog.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction

# Three well-known primes near 1e9; products of two fit comfortably in 61 bits.
DEFAULT_PRIMES = (1_000_000_007, 1_000_000_009, 998_244_353)


def _to_fp(x, p: int) -> int:
    """Reduce an int or Fraction into F_p. Raises if a denominator is divisible by p
    (a bad prime for this entry) rather than silently returning a wrong residue."""
    if isinstance(x, int):
        return x % p
    num, den = x.numerator % p, x.denominator % p
    if den == 0:
        raise ValueError(f"prime {p} divides a denominator; choose a different prime.")
    return num * pow(den, p - 2, p) % p


def rank_mod_p(columns: Iterable[Mapping[int, object]], p: int) -> int:
    """GF(p) column rank by Gaussian elimination over Python ints (coefficients stay in
    [0, p), so no Fraction blow-up)."""
    pivots: dict[int, dict[int, int]] = {}      # leading row -> reduced column
    rank = 0
    for col in columns:
        v = {r: m for r, c in col.items() if (m := _to_fp(c, p))}
        while v:
            lead = max(v)
            piv = pivots.get(lead)
            if piv is None:
                pivots[lead] = v
                rank += 1
                break
            factor = v[lead] * pow(piv[lead], p - 2, p) % p
            for r, c in piv.items():
                nv = (v.get(r, 0) - factor * c) % p
                if nv:
                    v[r] = nv
                else:
                    v.pop(r, None)
    return rank


def rational_rank_multimodular(columns, primes: tuple[int, ...] = DEFAULT_PRIMES) -> int:
    """Rank over ℚ as the max of the ranks mod several primes (bad primes only lower it)."""
    cols = list(columns)
    return max(rank_mod_p(cols, p) for p in primes)


def rational_homology_multimodular(cx, primes: tuple[int, ...] = DEFAULT_PRIMES) -> dict[int, int]:
    """Rational homology dimensions via multimodular ranks -- identical to
    rational_reduce.rational_homology, faster on large complexes."""
    ranks = {n: rational_rank_multimodular(cx.differential(n), primes) for n in cx.degrees()}
    result: dict[int, int] = {}
    for n in cx.degrees():
        h = cx.dim(n) - ranks.get(n, 0) - ranks.get(n - 1, 0)
        if h < 0:
            raise RuntimeError(f"negative Betti number at degree {n}: not a valid complex.")
        if h:
            result[n] = h
    return result
