"""Tests for the grid-diagram model and gradings (engine Phase 6).

The model's correctness is pinned by an already-validated invariant: tracing the KnotInfo
markers into O/X and computing the Maslov/Alexander gradings must make the generators'
graded Euler characteristic equal (1 - t)^{n-1} * Delta_K(t) (up to a unit), where Delta_K
is our own native Alexander polynomial. So this checks the grid labelling and both gradings
end to end, before any differential exists.
"""
from collections import defaultdict
from math import comb

import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.floer import alexander_euler_characteristic
from tetradrome.engines.floer.grid import GridDiagram
from tetradrome.errors import TetradromeError
from tetradrome.invariants.seifert import alexander_polynomial, seifert_matrix_from_braid

KNOTS = ["3_1", "4_1", "5_1", "5_2"]   # grid sizes <= 7, so n! stays cheap


def _native_alexander(name):
    return alexander_polynomial(seifert_matrix_from_braid(ki.braid_word(name)))


def _poly_mul(a, b):
    out = defaultdict(int)
    for da, ca in a.items():
        for db, cb in b.items():
            out[da + db] += ca * cb
    return out


def _one_minus_t_pow(k):
    return {i: (-1) ** i * comb(k, i) for i in range(k + 1)}


def _normalize(coeffs_by_deg):
    """Canonical form up to multiplication by +/- t^k: shift to start at degree 0 and make
    the lowest coefficient positive."""
    items = {d: c for d, c in coeffs_by_deg.items() if c}
    lo = min(items)
    seq = [items.get(d + lo, 0) for d in range(max(items) - lo + 1)]
    return tuple(-c for c in seq) if seq[0] < 0 else tuple(seq)


def test_from_markers_traces_valid_permutations():
    for name in KNOTS:
        grid = GridDiagram.from_knotinfo(name)
        assert sorted(grid.O) == list(range(grid.n))
        assert sorted(grid.X) == list(range(grid.n))
        assert all(grid.O[i] != grid.X[i] for i in range(grid.n))


@pytest.mark.parametrize("name", KNOTS)
def test_euler_characteristic_reproduces_alexander(name):
    grid = GridDiagram.from_knotinfo(name)
    chi = alexander_euler_characteristic(grid)
    delta = {i: c for i, c in enumerate(_native_alexander(name))}
    expected = _poly_mul(_one_minus_t_pow(grid.n - 1), delta)
    assert _normalize(chi) == _normalize(expected)


def test_invalid_grids_raise():
    with pytest.raises(TetradromeError):       # O not a permutation
        GridDiagram([0, 0], [1, 0])
    with pytest.raises(TetradromeError):       # O and X overlap in a row
        GridDiagram([0, 1], [0, 1])
    with pytest.raises(TetradromeError):       # odd marker count
        GridDiagram.from_markers([[1, 1], [1, 2], [2, 1]])
