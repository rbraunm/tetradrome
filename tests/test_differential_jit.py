# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Agreement: the numba-JIT differential tier equals the reference differential, exhaustively.

The reference ``differential`` in ``differential.py`` is the canonical definition; this pins
the jit tier to it bit-for-bit over EVERY generator of each grid (an exhaustive sweep at a size
is a proof of equivalence at that size). The jit tier reorganizes the F2 fold by unordered row
pair, so the comparison reconstructs each target permutation from the surviving pairs and checks
the resulting ``{target: 1}`` map equals the reference's. Default runs cover the staircase grids
and the n <= 8 KnotInfo roster (real marker layouts); ``--heavy`` adds the larger staircases.
"""
from __future__ import annotations

import math

import pytest

from tetradrome.engines.floer.grid import GridDiagram, staircase_grid
from tetradrome.engines.floer.generation import _unrank
from tetradrome.engines.floer.differential import differential
from tetradrome.engines.floer.differential_jit import HAVE_NUMPY, differential_block

pytestmark = pytest.mark.skipif(not HAVE_NUMPY, reason="the jit differential tier needs numpy")


def _roster_params():
    """The n <= 8 roster as params, or one skip param if KnotInfo is unavailable -- resolved at
    collection so the suite stays importable without the optional backend (like the GPU skips)."""
    try:
        from tetradrome.engines.floer import floer_roster
        from tetradrome.errors import BackendUnavailable
    except Exception:  # pragma: no cover - import wiring
        return [pytest.param(None, id="knotinfo-unavailable", marks=pytest.mark.skip(
            reason="KnotInfo backend unavailable"))]
    try:
        return [pytest.param(name, id=f"{name}_n{n}") for name, n in floer_roster(8)]
    except BackendUnavailable:
        return [pytest.param(None, id="knotinfo-unavailable", marks=pytest.mark.skip(
            reason="KnotInfo backend unavailable"))]


def _assert_exhaustive_agreement(grid):
    import numpy as np

    n = grid.n
    total = math.factorial(n)
    states = np.array([_unrank(k, n) for k in range(total)], dtype=np.int64)
    out_pairs, out_counts = differential_block(states, grid.O, grid.X)
    assert out_counts.shape == (total,)
    for index in range(total):
        sigma = tuple(int(value) for value in states[index])
        reconstructed = {}
        for slot in range(int(out_counts[index])):
            code = int(out_pairs[index, slot])
            i, j = divmod(code, n)
            target = list(sigma)
            target[i], target[j] = sigma[j], sigma[i]
            reconstructed[tuple(target)] = 1
        assert reconstructed == differential(grid, sigma), (n, sigma)


@pytest.mark.parametrize("n", range(2, 9))
def test_differential_jit_matches_reference_staircase(n):
    _assert_exhaustive_agreement(staircase_grid(n))


@pytest.mark.parametrize("name", _roster_params())
def test_differential_jit_matches_reference_roster(name):
    if name is None:
        pytest.skip("KnotInfo backend unavailable")
    _assert_exhaustive_agreement(GridDiagram.from_knotinfo(name))


@pytest.mark.heavy
@pytest.mark.parametrize("n", (9, 10))
def test_differential_jit_matches_reference_staircase_large(n):
    _assert_exhaustive_agreement(staircase_grid(n))
