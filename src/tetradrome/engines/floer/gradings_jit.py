# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Numba-JIT grid gradings (engine Phase 6).

A typed-integer-array transcription of the Maslov and Alexander gradings defined
in ``gradings.py``: the same ``_sw(...)`` south-west pair counts and the same
``M_M(x) = sw(x,x) - sw(x,M) - sw(M,x) + sw(M,M) + 1`` relation, evaluated over a
whole block of generators in one pass. Two points keep it faithful and auditable:

* It is a literal translation of the definitions, not a different algorithm: the
  same nested enumeration of point pairs, integer arrays in place of Python
  tuples and the float coordinates. Generation gets the same answer; the speed is
  from removing CPython per-pair overhead, not from changing the maths.
* It uses doubled integer coordinates -- a generator point in column ``c`` sits at
  ``(2i, 2c)``, a marker at ``(2i+1, 2c+1)`` -- instead of the reference's
  half-integer ``(i, c+0.5)``. The strict south-west test ``px < qx and py < qy``
  has the identical truth value under both (``2i < 2k+1`` iff ``i <= k`` iff
  ``i < k+0.5``), so every count is equal, with no floating point anywhere. This
  is strictly more rigorous than the reference, which the agreement test pins.

The implementation is valid both as plain Python and under numba's nopython mode,
and numba/numpy are imported lazily, so the package imports (and the reference
gradings run) with neither installed; only a process that calls this tier
compiles it, and ``cache=True`` persists the compile across processes (mirrors
``reduce_f2_jit``). The reference ``gradings.py`` remains the canonical, readable
definition and the oracle this is checked against, exhaustively, in the tests.
"""
from __future__ import annotations

import importlib.util

from ...errors import BackendUnavailable

# Cheap presence checks that do NOT import (or compile) the optional deps.
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_NUMBA = importlib.util.find_spec("numba") is not None

np = None           # bound lazily on first call; numpy is an optional accel dependency
_kernel = None      # lazily bound to the compiled (or plain) implementation on first call


def _gradings_block_impl(states, o_col, x_col, n):
    """Per-generator ``(maslov, alexander)`` for a block of generators. njit-compatible.

    ``states`` is a ``(B, n)`` int array with ``states[b, i] = sigma(i)`` for generator
    ``b``; ``o_col[i] = 2*O[i] + 1`` and ``x_col[i] = 2*X[i] + 1`` are the doubled-odd
    marker columns. A generator point ``i`` sits at row ``2i`` column ``2*states[b,i]``
    (doubled-even); marker ``i`` at row ``2i+1`` column ``o_col[i]``/``x_col[i]``
    (doubled-odd). Returns ``(maslov[B], alexander[B])`` int arrays.
    """
    # Marker-vs-marker south-west counts are state-independent -- compute once.
    sw_oo = 0
    sw_xx = 0
    for a in range(n):
        ra = 2 * a + 1
        for b in range(n):
            rb = 2 * b + 1
            if ra < rb and o_col[a] < o_col[b]:
                sw_oo += 1
            if ra < rb and x_col[a] < x_col[b]:
                sw_xx += 1

    count = states.shape[0]
    maslov = np.empty(count, dtype=np.int64)
    alexander = np.empty(count, dtype=np.int64)
    for s in range(count):
        sw_gg = 0       # gen vs gen
        sw_go = 0       # gen vs O
        sw_og = 0       # O vs gen
        sw_gx = 0       # gen vs X
        sw_xg = 0       # X vs gen
        for i in range(n):
            gri = 2 * i
            gci = 2 * states[s, i]
            for m in range(n):
                grm = 2 * m
                gcm = 2 * states[s, m]
                if gri < grm and gci < gcm:
                    sw_gg += 1
                mr = 2 * m + 1
                oc = o_col[m]
                xc = x_col[m]
                if gri < mr and gci < oc:
                    sw_go += 1
                if gri < mr and gci < xc:
                    sw_gx += 1
                if mr < gri and oc < gci:
                    sw_og += 1
                if mr < gri and xc < gci:
                    sw_xg += 1
        m_o = sw_gg - sw_go - sw_og + sw_oo + 1
        m_x = sw_gg - sw_gx - sw_xg + sw_xx + 1
        maslov[s] = m_o
        alexander[s] = (m_o - m_x - (n - 1)) // 2
    return maslov, alexander


def _get_kernel():
    """Bind the kernel once: numba-compiled if numba is present, else the plain impl."""
    global _kernel
    if _kernel is None:
        if HAVE_NUMBA:
            import numba

            _kernel = numba.njit(cache=True)(_gradings_block_impl)
        else:
            _kernel = _gradings_block_impl
    return _kernel


def maslov_alexander_block(states, o_markers, x_markers):
    """``(maslov, alexander)`` int arrays for a block of generators ``states`` (``B x n``).

    ``states`` is any 2-D integer array-like of permutations; ``o_markers``/``x_markers``
    are the length-``n`` O/X column lists of the grid. Uses the numba-compiled kernel when
    numba is present, else the identical plain implementation.
    """
    global np
    if np is None:
        if not HAVE_NUMPY:
            raise BackendUnavailable(
                "the jit gradings tier needs numpy (pip install the 'jit' or 'accel' "
                "extra); the pure-Python gradings in gradings.py need no numpy."
            )
        import numpy as np

    states_arr = np.ascontiguousarray(states, dtype=np.int64)
    n = states_arr.shape[1]
    o_col = np.empty(n, dtype=np.int64)
    x_col = np.empty(n, dtype=np.int64)
    for i in range(n):
        o_col[i] = 2 * int(o_markers[i]) + 1
        x_col[i] = 2 * int(x_markers[i]) + 1
    return _get_kernel()(states_arr, o_col, x_col, n)
