# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Bit-packed F2 rank reducers -- the first acceleration tier (decision 0003).

Two flavours, both computing the same GF(2) column rank as the reference
`reduce_reference.f2_rank`, and both required to agree with it exactly (the agreement
discipline, design section 4):

- `f2_rank_bitint`: pure Python, each column a Python `int` used as a bit-vector. XOR is
  `^` (C-level on big ints), the leading row is `bit_length() - 1`. Zero dependencies,
  always available, and markedly faster than the set-based reference on dense columns.
- `f2_rank_words`: each column an array of uint64 words, parameterized by the array
  module `xp` -- `numpy` on CPU, `cupy` on GPU. The reduction is identical; only `xp`
  changes. This is the GPU-capable path (the CPU `numpy` run validates the exact code
  the GPU runs).

A column arrives in compressed-sparse-column form, the representation GradedComplex stores
and `GradedComplex.differential` returns: a pair ``(indices, indptr)`` over ``array('i')``
where column j is ``indices[indptr[j]:indptr[j+1]]``. The pure-Python reducer slices the
buffers directly; the packed reducers scatter them into word arrays with a single
vectorized numpy pass (`_pack_csc`). `pack_columns` builds the same CSC from a readable
column-by-column matrix for callers that have one (tests, ad-hoc use).

Both do column reduction: a column is XORed against the stored pivot for its leading
row until it either hits an empty pivot slot (new pivot, rank += 1) or reduces to zero.
The leading row strictly drops each step, so it terminates.
"""
from __future__ import annotations

from array import array


def pack_columns(columns) -> tuple[array, array]:
    """Pack a matrix given column-by-column (each column an iterable of row indices) into the
    CSC buffers ``(indices, indptr)`` over ``array('i')`` the reducers consume. The bridge
    from a readable per-column matrix to CSC; GradedComplex builds the same internally."""
    indices = array("i")
    indptr = array("i", [0])
    for col in columns:
        for r in col:
            indices.append(int(r))
        indptr.append(len(indices))
    return indices, indptr


def f2_rank_bitint(csc: tuple[array, array]) -> int:
    """GF(2) rank with each column packed into a Python int bit-vector. `csc` is the
    ``(indices, indptr)`` pair; column j is ``indices[indptr[j]:indptr[j+1]]``."""
    indices, indptr = csc
    pivots: dict[int, int] = {}  # leading row -> reduced column
    rank = 0
    for j in range(len(indptr) - 1):
        v = 0
        for r in indices[indptr[j]:indptr[j + 1]]:
            v |= 1 << r
        while v:
            lead = v.bit_length() - 1
            piv = pivots.get(lead)
            if piv is None:
                pivots[lead] = v
                rank += 1
                break
            v ^= piv
    return rank


def _pack_csc(indices: array, indptr: array, nrows: int, xp):
    """Pack CSC columns into an (n_cols, n_words) uint64 array over array module `xp`. The
    scatter is one vectorized numpy pass on the host -- `repeat` to label each entry with its
    column, then a single `bitwise_or.at` setting the bit -- and the finished matrix moves to
    the device in one transfer for the GPU path (per-element device assignment would dwarf
    the reduction)."""
    import numpy as np

    nwords = max(1, (nrows + 63) // 64)
    ptr = np.frombuffer(indptr, dtype=np.int32)
    ncols = ptr.shape[0] - 1
    if ncols <= 0:
        return xp.zeros((0, nwords), dtype=xp.uint64)
    host = np.zeros((ncols, nwords), dtype=np.uint64)
    idx = np.frombuffer(indices, dtype=np.int32)
    if idx.shape[0]:
        idx64 = idx.astype(np.int64)
        col_of = np.repeat(np.arange(ncols, dtype=np.int64), np.diff(ptr.astype(np.int64)))
        np.bitwise_or.at(
            host, (col_of, idx64 >> 6), np.uint64(1) << (idx64 & np.int64(63)).astype(np.uint64)
        )
    return xp.asarray(host)


def _leading_row(v, xp) -> int:
    """Highest set bit of a packed column `v` (a 1-D uint64 array), or -1 if zero."""
    nz = xp.nonzero(v)[0]
    if nz.size == 0:
        return -1
    top = int(nz[-1])
    return top * 64 + (int(v[top]).bit_length() - 1)


def f2_rank_words(csc: tuple[array, array], nrows: int, xp) -> int:
    """GF(2) rank with CSC columns packed into uint64 word arrays over array module `xp`
    (`numpy` for CPU, `cupy` for GPU). Same algorithm as `f2_rank_bitint`; the XOR is a
    vectorized word-array operation, so it carries to the GPU unchanged."""
    indices, indptr = csc
    mat = _pack_csc(indices, indptr, nrows, xp)
    pivots: dict[int, object] = {}  # leading row -> reduced packed column
    rank = 0
    for k in range(mat.shape[0]):
        v = mat[k].copy()
        while True:
            lead = _leading_row(v, xp)
            if lead < 0:
                break
            piv = pivots.get(lead)
            if piv is None:
                pivots[lead] = v
                rank += 1
                break
            v = xp.bitwise_xor(v, piv)
    return rank


def f2_rank_dense(csc: tuple[array, array], nrows: int, xp) -> int:
    """GF(2) rank by vectorized dense row reduction over array module `xp`.

    Unlike `f2_rank_words`, which syncs once per elimination step to find a leading bit,
    this eliminates a whole pivot column in one vectorized XOR across all affected rows and
    syncs only once per column (the pivot search). That trades memory (a dense uint8
    matrix) for far fewer host round-trips, which is what the GPU wants -- the batched
    kernel for `packed-gpu`. The numpy run validates the exact code cupy executes.

    The matrix is assembled on the host -- a single vectorized scatter from the CSC buffers
    -- and moved to the device in one transfer: per-element assignment on a GPU array is one
    kernel launch each, so building it on the device would dwarf the reduction.
    """
    import numpy as np

    indices, indptr = csc
    ptr = np.frombuffer(indptr, dtype=np.int32)
    ncols = ptr.shape[0] - 1
    if ncols <= 0 or nrows == 0:
        return 0
    host = np.zeros((nrows, ncols), dtype=np.uint8)
    idx = np.frombuffer(indices, dtype=np.int32)
    if idx.shape[0]:
        col_of = np.repeat(np.arange(ncols, dtype=np.int64), np.diff(ptr.astype(np.int64)))
        host[idx.astype(np.int64), col_of] = 1
    mat = xp.asarray(host)
    rank = 0
    prow = 0
    for c in range(ncols):
        nz = xp.nonzero(mat[prow:, c])[0]
        if nz.size == 0:
            continue
        r = int(prow + nz[0])
        if r != prow:
            mat[[r, prow]] = mat[[prow, r]]
        col_bits = mat[:, c].copy()
        col_bits[prow] = 0
        rows = xp.nonzero(col_bits)[0]
        if rows.size:
            mat[rows] ^= mat[prow]
        rank += 1
        prow += 1
        if prow >= nrows:
            break
    return rank
