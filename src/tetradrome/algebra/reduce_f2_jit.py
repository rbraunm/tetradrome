# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Numba-JIT bit-packed F2 reducer (engine Phase 5).

The reduction logic lives in `_f2_rank_packed_impl`, written to be valid both as plain
numpy and under numba's nopython mode: explicit loops over a uint64 word matrix, an
integer pivot table indexed by leading row, in-place column XOR. When numba is installed
the function is compiled with `njit` on first use; otherwise it runs as ordinary Python,
so `f2_rank_jit` is correct everywhere and merely faster with numba.

Both numpy and numba are imported lazily, only when the jit reducer is first called -- not
at import time. For numpy that keeps the package importable, and the pure-Python `reference`
and `bitint` tiers usable, with no numpy installed (it is an optional accel dependency);
numpy is required only when this tier actually runs. The lazy numba import additionally
avoids paying its cost under multiprocessing `spawn` in workers that run a *different*
backend -- only workers that run the jit tier compile it.
"""
from __future__ import annotations

import importlib.util

from ..errors import BackendUnavailable
from .reduce_f2_packed import _pack_csc

# Cheap presence checks that do NOT import (or compile) the optional deps.
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_NUMBA = importlib.util.find_spec("numba") is not None

np = None           # bound lazily on first call; numpy is an optional accel dependency
_reducer = None     # lazily bound to the compiled (or plain) implementation on first call


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


def _get_reducer():
    """Bind the reducer once: numba-compiled if numba is present, else the plain impl."""
    global _reducer
    if _reducer is None:
        if HAVE_NUMBA:
            import numba

            _reducer = numba.njit(cache=True)(_f2_rank_packed_impl)
        else:
            _reducer = _f2_rank_packed_impl
    return _reducer


def f2_rank_jit(csc, nrows: int) -> int:
    """GF(2) rank via the (numba-compiled if available) packed reducer. `csc` is the
    ``(indices, indptr)`` pair GradedComplex.differential returns."""
    global np
    if np is None:
        if not HAVE_NUMPY:
            raise BackendUnavailable(
                "the jit/packed reducer needs numpy (pip install the 'jit' or 'accel' "
                "extra); the pure-Python 'reference' and 'bitint' backends need no numpy."
            )
        import numpy as np
    indices, indptr = csc
    mat = _pack_csc(indices, indptr, nrows, np)
    if mat.shape[0] == 0:
        return 0
    return int(_get_reducer()(mat, mat.shape[1]))
