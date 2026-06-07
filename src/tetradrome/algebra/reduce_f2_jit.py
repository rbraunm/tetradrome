"""Numba-JIT bit-packed F2 reducer (engine Phase 5).

The reduction logic lives in `_f2_rank_packed_impl`, written to be valid both as plain
numpy and under numba's nopython mode: explicit loops over a uint64 word matrix, an
integer pivot table indexed by leading row, in-place column XOR. When numba is installed
the function is wrapped with `njit` for native speed; otherwise it runs as ordinary
Python, so `f2_rank_jit` is correct everywhere and merely faster with numba.

Writing it this way means the algorithm is validated in environments without numba (the
un-compiled function is exercised against the reference), and numba only changes how fast
the identical code runs -- the agreement discipline holds regardless.
"""
from __future__ import annotations

import numpy as np

from .reduce_f2_packed import _pack

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def _f2_rank_packed_impl(mat, nwords):
    """GF(2) column rank of a packed (n_cols, nwords) uint64 matrix. njit-compatible."""
    n_cols = mat.shape[0]
    nbits = nwords * 64
    max_piv = n_cols if n_cols < nbits else nbits
    pivot_of_row = np.full(nbits, -1, dtype=np.int64)   # leading row -> pivot slot, or -1
    pivots = np.zeros((max_piv, nwords), dtype=np.uint64)
    v = np.zeros(nwords, dtype=np.uint64)
    one = np.uint64(1)
    rank = 0
    for k in range(n_cols):
        for w in range(nwords):
            v[w] = mat[k, w]
        while True:
            lead = -1
            for w in range(nwords - 1, -1, -1):
                if v[w] != 0:
                    word = v[w]
                    b = 63
                    while b >= 0:
                        if (word >> np.uint64(b)) & one:
                            break
                        b -= 1
                    lead = w * 64 + b
                    break
            if lead < 0:
                break
            p = pivot_of_row[lead]
            if p < 0:
                pivot_of_row[lead] = rank
                for w in range(nwords):
                    pivots[rank, w] = v[w]
                rank += 1
                break
            for w in range(nwords):
                v[w] = v[w] ^ pivots[p, w]
    return rank


_f2_rank_packed = numba.njit(cache=True)(_f2_rank_packed_impl) if HAVE_NUMBA else _f2_rank_packed_impl


def f2_rank_jit(columns, nrows: int) -> int:
    """GF(2) rank via the (numba-compiled if available) packed reducer."""
    mat = _pack(columns, nrows, np)
    if mat.shape[0] == 0:
        return 0
    return int(_f2_rank_packed(mat, mat.shape[1]))
