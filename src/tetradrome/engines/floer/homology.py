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

from ...algebra import GradedComplex, f2_homology, parallel_f2_homology
from ...errors import UnvalidatedResult
from .differential import differential
from .gradings import alexander, maslov


def grid_complexes(grid) -> dict:
    """The generation step: build the per-Alexander-grading F2 complexes ``{A: GradedComplex}``.

    Within an Alexander grading the complex is graded by degree = -Maslov (so the back end's
    degree-raising differential matches), and the bigraded differential preserves Alexander, so
    every target lands in the same grading. This is the n! step -- it dominates for large grids
    and is what a scaling study measures separately from the reduction.
    """
    by_alexander: dict = defaultdict(list)
    for state in grid.generators():
        by_alexander[alexander(grid, state)].append(state)

    complexes: dict = {}
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
        complexes[a_grading] = GradedComplex(dict(dims), columns)
    return complexes


def reduce_complexes(complexes: dict, *, backend: str = "bitint",
                     workers: int = 1, pin: bool = False) -> dict:
    """Reduce ``{A: GradedComplex}`` to ``{(Maslov, Alexander): dimension}``.

    The per-grading complexes are independent, so with ``workers > 1`` they reduce across
    processes (``parallel_f2_homology``, CPU backends only, optional NUMA ``pin``). Every
    backend returns the identical answer as the reference (Phase 5 agreement discipline).
    """
    if workers > 1:
        homologies = parallel_f2_homology(complexes, backend=backend, workers=workers, pin=pin)
    else:
        homologies = {a: f2_homology(cx, backend=backend) for a, cx in complexes.items()}

    poincare: dict = defaultdict(int)
    for a_grading, homology in homologies.items():
        for degree, dim in homology.items():
            poincare[(-degree, a_grading)] += dim
    return {key: value for key, value in poincare.items() if value}


def grid_poincare(grid, *, backend: str = "bitint", workers: int = 1, pin: bool = False) -> dict:
    """Grid (hat) homology as ``{(Maslov, Alexander): dimension}`` over F2."""
    return reduce_complexes(
        grid_complexes(grid), backend=backend, workers=workers, pin=pin
    )


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


def hfk_hat(grid, *, backend: str = "bitint", workers: int = 1, pin: bool = False) -> dict:
    """HFK-hat as ``{(Maslov, Alexander): rank}``.

    Divides the grid Poincare polynomial by (1 + q^{-1} t^{-1})^{n-1} and verifies the
    quotient by reconstruction. With the grid in the standard chirality this matches
    KnotInfo directly.
    """
    grid_homology = grid_poincare(grid, backend=backend, workers=workers, pin=pin)
    quotient = grid_homology
    for _ in range(grid.n - 1):
        quotient = _divide_by_V_once(quotient)
    quotient = {key: value for key, value in quotient.items() if value}
    if any(value < 0 for value in quotient.values()) or _tensor_V(quotient, grid.n - 1) != grid_homology:
        raise UnvalidatedResult(
            "grid homology did not factor as HFK-hat (x) V^{n-1}; the V-factor division failed."
        )
    return quotient


def seifert_genus(grid, *, backend: str = "bitint", workers: int = 1, pin: bool = False) -> int:
    """Seifert genus: the top Alexander grading carrying nonzero HFK-hat (genus detection)."""
    return max(a for _, a in hfk_hat(grid, backend=backend, workers=workers, pin=pin))
