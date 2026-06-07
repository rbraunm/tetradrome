"""The Rasmussen s-invariant, read off the quantum filtration on Lee homology.

For a knot, Lee homology is 2-dimensional over Q and concentrated in homological degree
0. The quantum filtration F^k C^0 (spanned by the generators of quantum degree >= k)
descends to a filtration of H^0, whose two jumps occur at the filtration levels of the
two Lee generators. Rasmussen proved those levels are s-1 and s+1, so s = s_min + 1.

The filtration dimension is pure linear algebra over Q:

    dim F^k H^0 = dim(Z n F^k) - dim(B n F^k),   Z = ker d^0,  B = im d^(-1),

with
    dim(Z n F^k) = #{C^0 gens, q >= k} - rank(d^0 restricted to those source columns),
    dim(B n F^k) = rank(d^(-1)) - rank(d^(-1) with rows of q >= k dropped),

each rank computed by the rational reference reducer on a q-filtered submatrix. The two
k where this drops (2 -> 1 -> 0) are s_min and s_max; we check s_max = s_min + 2 as
Rasmussen's theorem demands, and return s_min + 1.

Chirality: this is the s-invariant of the given diagram. Our Khovanov/Lee gradings are
the mirror of KnotInfo's table convention (pinned in Phase 2c), and s(mirror) = -s(K),
so validation compares against -(KnotInfo's stored value); see the tests.
"""
from __future__ import annotations

from ...algebra import rational_homology, rational_rank
from ...diagrams.model import PDCode
from .lee import lee_complex_graded


def rasmussen_s(pd: PDCode) -> int:
    """The Rasmussen s-invariant of the knot with diagram `pd`. Raises if the Lee
    homology is not 2-dimensional in degree 0 (i.e. the input is not a knot) or if the
    filtration gap is not 2 (which would mean the construction is wrong)."""
    if not pd:
        return 0  # the unknot

    cx, qdeg = lee_complex_graded(pd)
    lee = rational_homology(cx)              # verifies d^2 = 0 over Q
    if lee != {0: 2}:
        raise RuntimeError(
            f"Lee homology is {lee}, not 2-dimensional in homological degree 0 -- the "
            f"s-invariant computation assumes a knot."
        )

    q0 = qdeg.get(0, [])                     # quantum degree of each C^0 generator
    d_minus = cx.differential(-1)            # columns C^(-1) -> C^0; rows index C^0
    d_zero = cx.differential(0)              # columns C^0 -> C^1; column pos indexes C^0
    rank_b = rational_rank(d_minus)          # dim B = im d^(-1)

    def filt_dim(k: int) -> int:
        """dim F^k H^0, the part of Lee homology representable in quantum degree >= k."""
        n_ge = sum(1 for q in q0 if q >= k)
        z_columns = [col for col, q in zip(d_zero, q0) if q >= k]
        z_dim = n_ge - rational_rank(z_columns)
        b_low = [{r: c for r, c in col.items() if q0[r] < k} for col in d_minus]
        b_dim = rank_b - rational_rank(b_low)
        return z_dim - b_dim

    levels = sorted(set(q0))
    s_min = max(k for k in levels if filt_dim(k) == 2)
    s_max = max(k for k in levels if filt_dim(k) == 1)
    if s_max != s_min + 2:
        raise RuntimeError(
            f"Lee quantum filtration gap is {s_max - s_min}, expected 2 "
            f"(Rasmussen: the two generators sit at s-1 and s+1)."
        )
    return s_min + 1
