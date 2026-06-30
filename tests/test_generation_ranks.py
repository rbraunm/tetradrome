# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Agreement: the lexicographic ranking that gives generators their integer identity is exact.

``_rank`` must invert ``_unrank`` (so a generator's rank round-trips), and the jit
``target_ranks_block`` must return ``_rank`` of the transposed permutation for every surviving
pair -- this is the identity the scheduled merge keys on, so an error here would corrupt the
complexes (the scheduler's serial-vs-grid_complexes test would also catch it; this pins the cause
directly). Ranking is pure permutation arithmetic, independent of the markers, so the staircase
grids exhaust the permutation space and cover it.
"""
from __future__ import annotations

import math

import pytest

from tetradrome.engines.floer.grid import staircase_grid
from tetradrome.engines.floer.generation import _rank, _unrank
from tetradrome.engines.floer.differential_jit import (
    HAVE_NUMPY,
    differential_block,
    target_ranks_block,
    unrank_block,
)


@pytest.mark.parametrize("n", range(2, 9))
def test_rank_inverts_unrank(n):
    for k in range(math.factorial(n)):
        assert _rank(_unrank(k, n)) == k


@pytest.mark.skipif(not HAVE_NUMPY, reason="the jit tier needs numpy")
@pytest.mark.parametrize("n", range(2, 9))
def test_unrank_block_matches_unrank(n):
    # The vectorized state build must reproduce _unrank bit-for-bit over a whole slice: this is the
    # generation slice's input, pinned directly (the scheduler's grid_complexes test catches it too,
    # but this isolates the cause). Pure permutation arithmetic, so no grid is needed.
    import numpy as np

    total = math.factorial(n)
    got = unrank_block(0, total, n)
    expected = np.array([_unrank(k, n) for k in range(total)], dtype=np.int64)
    assert got.shape == (total, n)
    assert np.array_equal(got, expected)


@pytest.mark.skipif(not HAVE_NUMPY, reason="the jit tier needs numpy")
def test_unrank_block_contiguous_slice_and_empty():
    # Generation requests contiguous slices [start, stop); a start past zero must match _unrank, and
    # an empty slice must yield no rows (the start >= stop case generation guards before calling).
    import numpy as np

    n = 7
    start, stop = 1234, 4321
    got = unrank_block(start, stop, n)
    expected = np.array([_unrank(k, n) for k in range(start, stop)], dtype=np.int64)
    assert np.array_equal(got, expected)
    assert unrank_block(9, 9, n).shape == (0, n)


@pytest.mark.heavy
def test_rank_inverts_unrank_n9():
    for k in range(math.factorial(9)):
        assert _rank(_unrank(k, 9)) == k


@pytest.mark.skipif(not HAVE_NUMPY, reason="the jit tier needs numpy")
@pytest.mark.parametrize("n", range(2, 9))
def test_target_ranks_match_rank_of_transposition(n):
    import numpy as np

    grid = staircase_grid(n)
    states = np.array([_unrank(k, n) for k in range(math.factorial(n))], dtype=np.int64)
    out_pairs, out_counts = differential_block(states, grid.O, grid.X)
    target_ranks = target_ranks_block(states, out_pairs, out_counts)
    for s in range(states.shape[0]):
        sigma = [int(value) for value in states[s]]
        for slot in range(int(out_counts[s])):
            code = int(out_pairs[s, slot])
            i, j = divmod(code, n)
            transposed = list(sigma)
            transposed[i], transposed[j] = sigma[j], sigma[i]
            assert int(target_ranks[s, slot]) == _rank(tuple(transposed))
