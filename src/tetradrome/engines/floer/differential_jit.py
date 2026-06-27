# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Numba-JIT grid (knot Floer) differential (engine Phase 6).

A typed-integer-array transcription of ``differential`` in ``differential.py``: the same
empty-rectangle enumeration, the same emptiness and marker-avoidance tests, the same F2
reduction -- reorganized by unordered row pair so the parity is exact and allocation-free.

The reorganization is an identity, not a change of maths. ``differential`` enumerates ordered
row pairs ``(i, j)`` and yields, for each empty marker-avoiding rectangle, the target obtained
by transposing the points in rows ``i`` and ``j``. That target depends only on the unordered
pair ``{i, j}`` (transposing ``i, j`` and transposing ``j, i`` give the same permutation), and
distinct unordered pairs give distinct targets, so no two pairs ever collide on a target. The
two ordered rectangles of ``{i, j}`` -- ``R(i, j)`` and ``R(j, i)`` -- therefore contribute to
exactly one target, and the F2 count of that target is ``valid(i, j) + valid(j, i) (mod 2)``,
i.e. it survives iff exactly one of the two rectangles is empty and marker-avoiding. So the
differential is ``{ transpose(sigma, i, j) : i < j, valid(i, j) != valid(j, i) }`` -- the same
rectangles and tests as the reference, with the modulo-2 fold done per pair instead of through
a global counter.

``valid(a, b)`` is the reference's two checks verbatim, in modular integer arithmetic instead
of Python sets: a state point lies in the open rectangle iff its row is an interior row and its
column offset from ``sigma[a]`` is in ``[1, width-1]``; a marker lies in the (half-open) cell
box iff ``(row - a) mod n < height`` and ``(col - sigma[a]) mod n < width``.

The kernel is valid both as plain Python and under numba nopython; numba/numpy import lazily,
so the package imports (and the reference differential runs) with neither, only a process that
runs the tier compiles it, and ``cache=True`` persists the compile (mirrors ``reduce_f2_jit``).
The reference ``differential.py`` stays the canonical definition and the oracle the agreement
test pins this against, exhaustively.
"""
from __future__ import annotations

import importlib.util

from ...errors import BackendUnavailable

# Cheap presence checks that do NOT import (or compile) the optional deps.
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_NUMBA = importlib.util.find_spec("numba") is not None

np = None               # bound lazily on first call; numpy is an optional accel dependency
_rect_valid = None      # njit (or plain) helper, bound before the kernel that calls it
_kernel = None          # lazily bound block kernel


def _rect_valid_impl(states, s, a, b, n, marker_rows, marker_cols, marker_count):
    """Is the toroidal rectangle ``R(a, b)`` out of generator ``states[s]`` empty of other
    state points and free of any avoided marker? The reference's two tests, in integer form.

    ``R(a, b)`` has its south-west corner on the point in row ``a``, rises ``height`` rows to
    row ``b`` and extends ``width`` columns right to column ``states[s, b]`` (all toroidal).
    """
    corner_col = states[s, a]
    height = (b - a) % n
    width = (states[s, b] - corner_col) % n
    # Empty of other state points: no interior row's point falls in an interior column.
    for t in range(1, height):
        row = (a + t) % n
        offset = (states[s, row] - corner_col) % n
        if offset >= 1 and offset <= width - 1:
            return False
    # Marker-avoiding: no avoided marker lies in the half-open cell box
    # rows [a, a+height) x columns [corner_col, corner_col+width).
    for m in range(marker_count):
        if ((marker_rows[m] - a) % n < height
                and (marker_cols[m] - corner_col) % n < width):
            return False
    return True


def _differential_pairs_impl(states, marker_rows, marker_cols, marker_count, n,
                             out_pairs, out_counts):
    """For each generator in ``states`` (``B x n``), fill ``out_pairs[s, :out_counts[s]]`` with
    the surviving unordered row pairs, packed as ``i*n + j`` (``i < j``): the pairs whose two
    toroidal rectangles disagree on validity, i.e. the F2 differential targets by transposition.
    """
    count = states.shape[0]
    for s in range(count):
        written = 0
        for i in range(n):
            for j in range(i + 1, n):
                valid_ij = _rect_valid(states, s, i, j, n,
                                       marker_rows, marker_cols, marker_count)
                valid_ji = _rect_valid(states, s, j, i, n,
                                       marker_rows, marker_cols, marker_count)
                if valid_ij != valid_ji:
                    out_pairs[s, written] = i * n + j
                    written += 1
        out_counts[s] = written


def _bind():
    """Bind the helper then the kernel once: numba-compiled if numba is present (the kernel
    references the helper as a module global, so it must be bound first), else the plain impls."""
    global _rect_valid, _kernel
    if _kernel is None:
        if HAVE_NUMBA:
            import numba

            _rect_valid = numba.njit(cache=True)(_rect_valid_impl)
            _kernel = numba.njit(cache=True)(_differential_pairs_impl)
        else:
            _rect_valid = _rect_valid_impl
            _kernel = _differential_pairs_impl
    return _kernel


def differential_block(states, o_markers, x_markers):
    """Surviving transposition pairs of the (hat) differential for a block of generators.

    Returns ``(out_pairs, out_counts)``: for generator ``s``, ``out_pairs[s, :out_counts[s]]``
    holds the packed ``i*n + j`` (``i < j``) of each surviving pair; transposing rows ``i, j``
    of ``states[s]`` gives a target with F2 coefficient 1. Avoids every O and X marker, matching
    ``differential``. Uses the numba-compiled kernel when numba is present, else the plain impl.
    """
    global np
    if np is None:
        if not HAVE_NUMPY:
            raise BackendUnavailable(
                "the jit differential tier needs numpy (pip install the 'jit' or 'accel' "
                "extra); the pure-Python differential in differential.py needs no numpy."
            )
        import numpy as np

    states_arr = np.ascontiguousarray(states, dtype=np.int64)
    n = states_arr.shape[1]
    marker_count = 2 * n
    marker_rows = np.empty(marker_count, dtype=np.int64)
    marker_cols = np.empty(marker_count, dtype=np.int64)
    for r in range(n):
        marker_rows[r] = r
        marker_cols[r] = int(o_markers[r])
        marker_rows[n + r] = r
        marker_cols[n + r] = int(x_markers[r])
    max_pairs = n * (n - 1) // 2
    out_pairs = np.empty((states_arr.shape[0], max_pairs), dtype=np.int64)
    out_counts = np.empty(states_arr.shape[0], dtype=np.int64)
    _bind()(states_arr, marker_rows, marker_cols, marker_count, n, out_pairs, out_counts)
    return out_pairs, out_counts


def differential_jit(grid, sigma) -> dict:
    """Bigraded (hat) differential ``d(sigma)`` as ``{target: 1}`` over F2, via the jit tier.

    A drop-in for ``differential`` for one generator: reconstructs the target permutations from
    the surviving transposition pairs the kernel returns.
    """
    global np
    if np is None:
        if not HAVE_NUMPY:
            raise BackendUnavailable(
                "the jit differential tier needs numpy (pip install the 'jit' or 'accel' "
                "extra); the pure-Python differential in differential.py needs no numpy."
            )
        import numpy as np

    n = len(sigma)
    out_pairs, out_counts = differential_block(np.array([sigma], dtype=np.int64),
                                               grid.O, grid.X)
    base = list(sigma)
    result: dict = {}
    for index in range(int(out_counts[0])):
        code = int(out_pairs[0, index])
        i, j = divmod(code, n)
        target = list(base)
        target[i], target[j] = base[j], base[i]
        result[tuple(target)] = 1
    return result
