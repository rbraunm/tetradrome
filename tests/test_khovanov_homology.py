"""Validate native Khovanov homology against KnotInfo (Phase 2c).

KnotInfo stores the unreduced INTEGRAL Khovanov homology (with Z/2 torsion). The mod-2
Betti numbers follow by universal coefficients -- Khovanov is a cochain complex of free
abelian groups, so (cohomological direction)

    dim_F2 Kh^{i,j} = free_rank(i,j) + tor2(i,j) + tor2(i+1,j),

where tor2 counts the even-order torsion summands.

One wrinkle, diagnosed empirically: KnotInfo's Khovanov table is tabulated in the
opposite chirality from its own stored PD code (our Khovanov of KnotInfo's 3_1 PD
matches the Knot Atlas LEFT trefoil and is the exact mirror of KnotInfo's 3_1 table).
Our value is the correct Khovanov of the given diagram -- d^2 = 0 holds and the graded
Euler characteristic equals that diagram's Jones polynomial -- so we compare up to the
global mirror (i, j) -> (-i, -j), which held exactly across the small-knot table.
"""
import ast

import pytest

from tetradrome import knots
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines import khovanov

# <= 7 crossings: enhanced-state enumeration is exponential.
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]


def _knotinfo_mod2_unreduced(name: str) -> dict[tuple[int, int], int]:
    """KnotInfo's unreduced Khovanov over F2, via universal coefficients from the
    stored integral vector [torsion, multiplicity, i, j]."""
    vec = ast.literal_eval(ki.lookup(name)["khovanov_unreduced_integral_vector"])
    free: dict[tuple[int, int], int] = {}
    tor2: dict[tuple[int, int], int] = {}
    for torsion, mult, i, j in vec:
        if torsion == 0:
            free[(i, j)] = free.get((i, j), 0) + mult
        elif torsion % 2 == 0:
            tor2[(i, j)] = tor2.get((i, j), 0) + mult
    keys = set(free) | set(tor2) | {(i - 1, j) for (i, j) in tor2}
    out: dict[tuple[int, int], int] = {}
    for (i, j) in keys:
        d = free.get((i, j), 0) + tor2.get((i, j), 0) + tor2.get((i + 1, j), 0)
        if d:
            out[(i, j)] = d
    return out


def _mirror(table: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    return {(-i, -j): d for (i, j), d in table.items()}


@pytest.mark.parametrize("name", KNOTS)
def test_khovanov_matches_knotinfo(name):
    pd = knots.from_name(name).pd_code
    mine = khovanov.khovanov_homology(pd)
    assert mine == _mirror(_knotinfo_mod2_unreduced(name))


def test_unknot_khovanov():
    # Unreduced Khovanov of the unknot over F2: F2 in (0, +1) and (0, -1).
    assert khovanov.khovanov_homology(()) == {(0, 1): 1, (0, -1): 1}


def test_trefoil_explicit():
    # KnotInfo's 3_1 PD is the left-handed trefoil; its F2 Khovanov (Knot Atlas).
    pd = knots.from_name("3_1").pd_code
    assert khovanov.khovanov_homology(pd) == {
        (0, -1): 1,
        (0, -3): 1,
        (-2, -5): 1,
        (-2, -7): 1,
        (-3, -7): 1,
        (-3, -9): 1,
    }
