"""Scaling helpers for the grid engine (engine Phase 6).

Grid homology is bounded by the n! generators, in two separable costs: *generation*
(enumerating the permutations and computing each one's gradings and differential) and
*reduction* (the F2 linear algebra). Generation is embarrassingly parallel -- every generator
is independent -- so this module hands contiguous slices of the permutation space to worker
processes, each unranking its own slice (no input IPC) and returning per-state data the parent
merges. Because the slices are contiguous in lexicographic order, the merged positions match
the serial enumeration exactly, so the assembled complexes -- and therefore the homology --
are identical to ``grid_complexes`` (the Phase 5 agreement discipline, now across processes).

``staircase_grid`` builds a valid n x n diagram for any n (the unknot on an n-grid), so a
scaling sweep can push the generator count past the small tabulated knots; it isolates the
*generation* curve. Representative *reduction* cost needs real knot grids.
"""
from __future__ import annotations

import math
from collections import defaultdict
from multiprocessing import Pool

from ...algebra import GradedComplex
from .differential import differential
from .gradings import alexander, maslov
from .grid import GridDiagram


def staircase_grid(n: int) -> GridDiagram:
    """A valid n x n grid (O on the diagonal, X one step right) -- the unknot on an n-grid.
    Has the full n! generators, so it isolates generation scaling without a KnotInfo lookup."""
    if n < 2:
        raise ValueError("grid size must be at least 2.")
    return GridDiagram(list(range(n)), [(i + 1) % n for i in range(n)])


def _unrank(index: int, n: int) -> tuple:
    """The lexicographically index-th permutation of range(n) (factorial number system)."""
    available = list(range(n))
    perm = []
    for place in range(n - 1, -1, -1):
        factorial = math.factorial(place)
        choice, index = divmod(index, factorial)
        perm.append(available.pop(choice))
    return tuple(perm)


def _generate_slice(args):
    o_markers, x_markers, start, stop = args
    grid = GridDiagram(o_markers, x_markers)
    return [
        (state, maslov(grid, state), alexander(grid, state),
         tuple(differential(grid, state)))
        for state in (_unrank(k, grid.n) for k in range(start, stop))
    ]


def _assemble(records: list) -> dict:
    """Merge per-state (state, Maslov, Alexander, targets) records into {A: GradedComplex},
    assigning positions in record order (= global lexicographic order)."""
    by_alexander: dict = defaultdict(list)
    for state, m, a, targets in records:
        by_alexander[a].append((state, -m, targets))      # degree = -Maslov

    complexes: dict = {}
    for a_grading, group in by_alexander.items():
        degree = {state: d for state, d, _ in group}
        position: dict = {}
        dims: dict = defaultdict(int)
        for state, d, _ in group:
            position[state] = dims[d]
            dims[d] += 1
        columns = {d: [None] * dims[d] for d in dims}
        for state, d, targets in group:
            columns[d][position[state]] = frozenset(
                position[y] for y in targets if degree.get(y) == d + 1
            )
        complexes[a_grading] = GradedComplex(dict(dims), columns)
    return complexes


def parallel_grid_complexes(grid, workers: int) -> dict:
    """Generate ``{A: GradedComplex}`` across ``workers`` processes; identical to the serial
    ``grid_complexes``. Falls back to serial generation for one worker or a tiny grid."""
    from .homology import grid_complexes

    total = math.factorial(grid.n)
    if workers <= 1 or total < 2 * workers:
        return grid_complexes(grid)
    slices = [
        (grid.O, grid.X, total * w // workers, total * (w + 1) // workers)
        for w in range(workers)
    ]
    with Pool(processes=workers) as pool:
        chunks = pool.map(_generate_slice, slices, chunksize=1)
    return _assemble([record for chunk in chunks for record in chunk])
