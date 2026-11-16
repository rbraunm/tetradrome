# SPDX-License-Identifier: Apache-2.0

"""Imported by the scheduler's forkserver so the workers it forks inherit the block kernels bound.

A process binds a numba kernel on its first call, loading the cached compile; for the four grid
kernels together that costs about 0.6 s. Every scheduler worker is a fresh process, so without
this each slice job paid it: a generation slice of 4096 states at grid number 10 took 0.63 s in a
fresh process against 0.044 s once bound. Importing this module runs each kernel once, on the two
states of a 2 x 2 grid, with the argument types the slice jobs use.
"""
from __future__ import annotations

from .differential_jit import (
    HAVE_NUMBA,
    HAVE_NUMPY,
    differential_block,
    target_ranks_block,
    unrank_block,
)
from .gradings_jit import maslov_alexander_block

if HAVE_NUMPY and HAVE_NUMBA:
    _states = unrank_block(0, 2, 2)
    _o_markers, _x_markers = (0, 1), (1, 0)
    maslov_alexander_block(_states, _o_markers, _x_markers)
    _pairs, _counts = differential_block(_states, _o_markers, _x_markers)
    target_ranks_block(_states, _pairs, _counts)
