# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Agreement: the numba-JIT gradings tier equals the reference gradings, exhaustively.

The reference ``maslov``/``alexander`` in ``gradings.py`` are the canonical definitions;
this pins ``gradings_jit.maslov_alexander_block`` to them bit-for-bit over EVERY generator
of each grid. An exhaustive sweep at a size is a proof of equivalence at that size, not a
sample. Default runs cover the staircase grids and the n <= 8 KnotInfo roster (real marker
layouts); ``--heavy`` adds the larger staircases (the full permutation space at n = 9, 10)
on a capable box. Generation gradings depend only on the O/X markers, so the n <= 8 roster
covers marker-layout variety and the large staircases cover large-n permutation geometry.
"""
from __future__ import annotations

import math

import pytest

from tetradrome.engines.floer.grid import GridDiagram, staircase_grid
from tetradrome.engines.floer.generation import _unrank
from tetradrome.engines.floer.gradings import alexander, maslov
from tetradrome.engines.floer.gradings_jit import HAVE_NUMPY, maslov_alexander_block

pytestmark = pytest.mark.skipif(not HAVE_NUMPY, reason="the jit gradings tier needs numpy")


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

    total = math.factorial(grid.n)
    states = np.array([_unrank(k, grid.n) for k in range(total)], dtype=np.int64)
    maslov_jit, alexander_jit = maslov_alexander_block(states, grid.O, grid.X)
    assert maslov_jit.shape == (total,)
    assert alexander_jit.shape == (total,)
    for index in range(total):
        sigma = tuple(int(value) for value in states[index])
        assert int(maslov_jit[index]) == maslov(grid, sigma), (grid.n, sigma, "maslov")
        assert int(alexander_jit[index]) == alexander(grid, sigma), (grid.n, sigma, "alexander")


@pytest.mark.parametrize("n", range(2, 9))
def test_gradings_jit_matches_reference_staircase(n):
    _assert_exhaustive_agreement(staircase_grid(n))


@pytest.mark.parametrize("name", _roster_params())
def test_gradings_jit_matches_reference_roster(name):
    if name is None:
        pytest.skip("KnotInfo backend unavailable")
    _assert_exhaustive_agreement(GridDiagram.from_knotinfo(name))


@pytest.mark.heavy
@pytest.mark.parametrize("n", (9, 10))
def test_gradings_jit_matches_reference_staircase_large(n):
    _assert_exhaustive_agreement(staircase_grid(n))
