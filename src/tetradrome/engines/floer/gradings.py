"""Maslov and Alexander gradings of grid generators (engine Phase 6).

For point sets in the plane, write ``_sw(A, B)`` for the number of pairs (a in A, b in B)
with a strictly south-west of b. With the generator points on the integer lattice and the
O/X markers at cell centres, the Maslov grading relative to a marker set M is

    M_M(x) = _sw(x,x) - _sw(x,M) - _sw(M,x) + _sw(M,M) + 1

(the bilinear expansion of "_sw(x - M, x - M) + 1"), the Maslov grading is M_O, and the
Alexander grading is A(x) = (M_O(x) - M_X(x) - (n-1)) / 2 (Grid Homology, Ozsvath-
Stipsicz-Szabo, ch. 4).

Validated by the graded Euler characteristic: sum_x (-1)^{M(x)} t^{A(x)} reproduces
(1 - t)^{n-1} * Delta_K(t) up to a unit, i.e. the knot's Alexander polynomial -- so these
gradings and the O/X labelling are checked against an already-validated invariant before
any differential is built.
"""
from __future__ import annotations

from collections import defaultdict


def _sw(a, b) -> int:
    """Number of pairs (p in a, q in b) with p strictly south-west of q."""
    return sum(1 for px, py in a for qx, qy in b if px < qx and py < qy)


def _marker_points(grid):
    gen_o = [(i + 0.5, grid.O[i] + 0.5) for i in range(grid.n)]
    gen_x = [(i + 0.5, grid.X[i] + 0.5) for i in range(grid.n)]
    return gen_o, gen_x


def _maslov_rel(gen, markers) -> int:
    return _sw(gen, gen) - _sw(gen, markers) - _sw(markers, gen) + _sw(markers, markers) + 1


def maslov(grid, sigma) -> int:
    gen = [(i, sigma[i]) for i in range(grid.n)]
    o, _ = _marker_points(grid)
    return _maslov_rel(gen, o)


def alexander(grid, sigma) -> int:
    gen = [(i, sigma[i]) for i in range(grid.n)]
    o, x = _marker_points(grid)
    return (_maslov_rel(gen, o) - _maslov_rel(gen, x) - (grid.n - 1)) // 2


def alexander_euler_characteristic(grid) -> dict[int, int]:
    """Graded Euler characteristic ``{Alexander grading: signed generator count}``."""
    poly: dict[int, int] = defaultdict(int)
    for sigma in grid.generators():
        poly[alexander(grid, sigma)] += -1 if maslov(grid, sigma) % 2 else 1
    return {a: c for a, c in poly.items() if c}
