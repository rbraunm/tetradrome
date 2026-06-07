"""Grid homology, reduced to knot Floer homology HFK-hat (Phase 6).

The differential preserves the Alexander grading and lowers Maslov by one, so each Alexander
grading is a Maslov-graded F2 complex; we hand it to the shared back end with degree set to
-Maslov, so the back end's degree-raising differential matches. The grid (hat) homology of an
n x n diagram is HFK-hat(K) (x) V^{(n-1)} with V = F2_{(0,0)} (+) F2_{(-1,-1)}, so dividing
the grid Poincare polynomial by (1 + q^{-1} t^{-1})^{n-1} recovers HFK-hat; the quotient is
checked by reconstructing the product (fail loud on mismatch).

The flat KnotInfo marker list does not record which markers are O and which are X, so the
diagram fixes a chirality only up to the global O<->X swap; HFK-hat is therefore determined
up to mirror, (M, A) <-> (-M, -A). The tau invariant (later) pins chirality. The Seifert
genus is the top Alexander grading carrying nonzero HFK-hat (the genus-detection theorem).
"""
from __future__ import annotations

from collections import defaultdict

from ...algebra import GradedComplex, homology
from ...errors import UnvalidatedResult
from .differential import differential
from .gradings import alexander, maslov


def grid_poincare(grid) -> dict:
    """Grid (hat) homology as ``{(Maslov, Alexander): dimension}`` over F2."""
    by_alexander: dict = defaultdict(list)
    for state in grid.generators():
        by_alexander[alexander(grid, state)].append(state)

    poincare: dict = defaultdict(int)
    for a_grading, group in by_alexander.items():
        degree = {state: -maslov(grid, state) for state in group}
        position: dict = {}
        dims: dict = defaultdict(int)
        for state in group:
            position[state] = dims[degree[state]]
            dims[degree[state]] += 1
        columns = {d: [None] * dims[d] for d in dims}
        for state in group:
            d = degree[state]
            columns[d][position[state]] = frozenset(
                position[y] for y in differential(grid, state) if degree.get(y) == d + 1
            )
        for d, dim in homology(GradedComplex(dict(dims), columns)).items():
            poincare[(-d, a_grading)] += dim
    return {key: value for key, value in poincare.items() if value}


def _divide_by_V_once(p: dict) -> dict:
    """Divide a bigraded count by (1 + q^{-1} t^{-1}); solve from the top corner down."""
    if not p:
        return {}
    maslov_range = range(min(m for m, _ in p), max(m for m, _ in p) + 1)
    alex_range = range(min(a for _, a in p), max(a for _, a in p) + 1)
    quotient: dict = {}
    cells = [(m, a) for m in maslov_range for a in alex_range]
    for cell in sorted(cells, key=lambda c: c[0] + c[1], reverse=True):
        value = p.get(cell, 0) - quotient.get((cell[0] + 1, cell[1] + 1), 0)
        if value:
            quotient[cell] = value
    return quotient


def _tensor_V(h: dict, power: int) -> dict:
    p = dict(h)
    for _ in range(power):
        nxt: dict = defaultdict(int)
        for (m, a), c in p.items():
            nxt[(m, a)] += c
            nxt[(m - 1, a - 1)] += c
        p = {key: value for key, value in nxt.items() if value}
    return p


def hfk_hat(grid) -> dict:
    """HFK-hat as ``{(Maslov, Alexander): rank}`` (up to mirror).

    Divides the grid Poincare polynomial by (1 + q^{-1} t^{-1})^{n-1} and verifies the
    quotient by reconstruction.
    """
    grid_homology = grid_poincare(grid)
    quotient = grid_homology
    for _ in range(grid.n - 1):
        quotient = _divide_by_V_once(quotient)
    quotient = {key: value for key, value in quotient.items() if value}
    if any(value < 0 for value in quotient.values()) or _tensor_V(quotient, grid.n - 1) != grid_homology:
        raise UnvalidatedResult(
            "grid homology did not factor as HFK-hat (x) V^{n-1}; the V-factor division failed."
        )
    return quotient


def seifert_genus(grid) -> int:
    """Seifert genus: the top Alexander grading carrying nonzero HFK-hat (genus detection)."""
    return max(a for _, a in hfk_hat(grid))
