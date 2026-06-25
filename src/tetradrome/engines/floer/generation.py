# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Generation: building the per-Alexander-grading F2 complexes from a grid diagram (Phase 6).

Grid homology is bounded by its n! generators, and generation -- enumerating the permutations
and computing each one's gradings and differential -- is the dominant cost at large n, separate
from the reduction linear algebra. Generation is embarrassingly parallel: every generator is
independent. ``grid_complexes`` builds the complexes serially; ``_generate_slice`` does the same
for one contiguous lexicographic slice of the permutation space (unranking its own slice, no input
IPC), which is the primitive the compute scheduler fans out across processes and folds back in
slice order. Because the slices are contiguous, the merged positions match the serial enumeration
exactly, so the scheduled generation reproduces ``grid_complexes`` bit for bit (the agreement
discipline, across processes).

``grading_histogram`` walks the same permutation space but keeps only the per-grading counts
(no differential, no matrices), so it predicts the reducer's dense-matrix dimensions at sizes
far too large to build -- the input to the reduction-memory model.
"""
from __future__ import annotations

import math
from collections import defaultdict
from multiprocessing import Pool

from ...algebra import GradedComplex
from .differential import differential
from .gradings import alexander, maslov
from .grid import GridDiagram


def _unrank(index: int, n: int) -> tuple:
    """The lexicographically index-th permutation of range(n) (factorial number system)."""
    available = list(range(n))
    perm = []
    for place in range(n - 1, -1, -1):
        factorial = math.factorial(place)
        choice, index = divmod(index, factorial)
        perm.append(available.pop(choice))
    return tuple(perm)


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


def _generate_slice(args):
    o_markers, x_markers, start, stop = args
    grid = GridDiagram(o_markers, x_markers)
    return [
        (state, maslov(grid, state), alexander(grid, state),
         tuple(differential(grid, state)))
        for state in (_unrank(k, grid.n) for k in range(start, stop))
    ]


def _build_complexes(by_alexander: dict) -> dict:
    """Build {A: GradedComplex} from states already grouped by Alexander grading, assigning
    positions in group order (which the callers keep equal to global lexicographic order)."""
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


def _grading_slice(args):
    """Count generators per (Alexander, degree=-Maslov) over a slice -- gradings only, no
    differential, so this is O(slice) time and O(gradings) memory."""
    o_markers, x_markers, start, stop = args
    grid = GridDiagram(o_markers, x_markers)
    hist: dict = defaultdict(int)
    for k in range(start, stop):
        state = _unrank(k, grid.n)
        hist[(alexander(grid, state), -maslov(grid, state))] += 1
    return dict(hist)


def grading_histogram(grid, workers: int = 1) -> dict:
    """Generator count per ``(Alexander, degree)`` grading over all n! states, gradings only.

    Builds no differentials and no matrices, so it is memory-safe at any n (it stores only the
    O(n^2) grading counts). The dimensions it returns are what set the reducer's dense-matrix
    sizes, so this is the input to ``dense_reduction_bytes`` -- a memory projection that can be
    computed for sizes far too large to actually generate or reduce.
    """
    total = math.factorial(grid.n)
    if workers <= 1 or total < 2 * workers:
        return _grading_slice((grid.O, grid.X, 0, total))
    slices = [
        (grid.O, grid.X, total * w // workers, total * (w + 1) // workers)
        for w in range(workers)
    ]
    merged: dict = defaultdict(int)
    with Pool(processes=workers) as pool:
        for part in pool.imap_unordered(_grading_slice, slices, chunksize=1):
            for key, count in part.items():
                merged[key] += count
    return dict(merged)
