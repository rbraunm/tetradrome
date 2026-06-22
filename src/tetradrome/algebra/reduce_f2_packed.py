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

Both do column reduction: a column is XORed against the stored pivot for its leading
row until it either hits an empty pivot slot (new pivot, rank += 1) or reduces to zero.
The leading row strictly drops each step, so it terminates.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def f2_rank_bitint(columns: Iterable[Iterable[int]]) -> int:
    """GF(2) rank with each column packed into a Python int bit-vector."""
    pivots: dict[int, int] = {}     # leading row -> reduced column
    rank = 0
    for col in columns:
        v = 0
        for r in col:
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


def _pack(columns, nrows, xp):
    """Pack columns (sets of row indices) into an (n_cols, n_words) uint64 array."""
    nwords = max(1, (nrows + 63) // 64)
    rows = []
    for col in columns:
        words = [0] * nwords
        for r in col:
            words[r >> 6] |= 1 << (r & 63)
        rows.append(words)
    if not rows:
        return xp.zeros((0, nwords), dtype=xp.uint64)
    return xp.array(rows, dtype=xp.uint64)


def _leading_row(v, xp) -> int:
    """Highest set bit of a packed column `v` (a 1-D uint64 array), or -1 if zero."""
    nz = xp.nonzero(v)[0]
    if nz.size == 0:
        return -1
    top = int(nz[-1])
    return top * 64 + (int(v[top]).bit_length() - 1)


def f2_rank_words(columns: Sequence[Iterable[int]], nrows: int, xp) -> int:
    """GF(2) rank with columns packed into uint64 word arrays over array module `xp`
    (`numpy` for CPU, `cupy` for GPU). Same algorithm as `f2_rank_bitint`; the XOR is a
    vectorized word-array operation, so it carries to the GPU unchanged."""
    mat = _pack(columns, nrows, xp)
    pivots: dict[int, object] = {}      # leading row -> reduced packed column
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


def f2_rank_dense(columns: Sequence[Iterable[int]], nrows: int, xp) -> int:
    """GF(2) rank by vectorized dense row reduction over array module `xp`.

    Unlike `f2_rank_words`, which syncs once per elimination step to find a leading bit,
    this eliminates a whole pivot column in one vectorized XOR across all affected rows and
    syncs only once per column (the pivot search). That trades memory (a dense uint8
    matrix) for far fewer host round-trips, which is what the GPU wants -- the batched
    kernel for `packed-gpu`. The numpy run validates the exact code cupy executes.

    The matrix is assembled on the host and moved to the device in a single transfer:
    per-element assignment on a GPU array is one kernel launch each, so building it on the
    device would dwarf the reduction.
    """
    import numpy as np

    cols = list(columns)
    ncols = len(cols)
    if ncols == 0 or nrows == 0:
        return 0
    host = np.zeros((nrows, ncols), dtype=np.uint8)
    for j, col in enumerate(cols):
        for r in col:
            host[r, j] = 1
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
