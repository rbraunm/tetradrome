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

from ...algebra import GradedComplex
from .differential import differential
from .differential_jit import (
    HAVE_NUMBA,
    HAVE_NUMPY,
    differential_block,
    target_ranks_block,
)
from .gradings import alexander, maslov
from .gradings_jit import maslov_alexander_block
from .grid import GridDiagram

# Generation tier: the numba kernels when both numpy and numba are present (the fast path the
# scheduler fans out across processes), else the pure-Python reference below. grid_complexes,
# _grading_slice, and grading_histogram stay on the reference unconditionally -- grid_complexes
# as the canonical, auditable oracle, the grading walk because it must stay memory-safe at sizes
# too large to materialize. Capability is fixed per process, so this is selected up front, not a
# fallback that engages on failure (a selected jit path that errors surfaces, it does not retry).
_USE_JIT = HAVE_NUMBA and HAVE_NUMPY


def _unrank(index: int, n: int) -> tuple:
    """The lexicographically index-th permutation of range(n) (factorial number system)."""
    available = list(range(n))
    perm = []
    for place in range(n - 1, -1, -1):
        factorial = math.factorial(place)
        choice, index = divmod(index, factorial)
        perm.append(available.pop(choice))
    return tuple(perm)


def _rank(perm) -> int:
    """The lexicographic rank of permutation ``perm`` -- the inverse of ``_unrank`` (so
    ``_rank(_unrank(k, n)) == k``). The rank labels a generator by a single int, which is the
    generator identity the scheduled merge keys on."""
    n = len(perm)
    available = list(range(n))
    index = 0
    for position in range(n):
        choice = available.index(perm[position])
        index += choice * math.factorial(n - 1 - position)
        available.pop(choice)
    return index


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
    """One contiguous lexicographic slice of the generation step, as per-generator records
    ``(rank, maslov, alexander, target ranks)`` -- each generator and its differential targets
    identified by lexicographic rank (ints), not permutation tuples, so the merge keys on ints and
    no target permutation is materialized. Uses the numba kernels when available (``_USE_JIT``),
    else the pure-Python reference; both produce identical records (target order is irrelevant --
    consumed as a frozenset)."""
    o_markers, x_markers, start, stop = args
    grid = GridDiagram(o_markers, x_markers)
    if _USE_JIT:
        return _generate_slice_jit(grid, start, stop)
    return [
        (start + offset, maslov(grid, state), alexander(grid, state),
         tuple(_rank(target) for target in differential(grid, state)))
        for offset, state in enumerate(_unrank(k, grid.n) for k in range(start, stop))
    ]


def _generate_slice_jit(grid, start, stop):
    """``_generate_slice`` via the numba kernels: block gradings, the differential's surviving
    transposition pairs, and those targets' ranks -- three kernel calls over the whole slice, then
    integer records with no permutation materialized. A generator's own rank is its slice offset
    from ``start``. Equal to the reference path bit-for-bit -- the per-tier agreement tests and the
    scheduler's ``test_generation_graph_matches_serial`` (which folds these slices and compares to
    ``grid_complexes``) pin it."""
    if start >= stop:
        return []
    import numpy as np

    states = [_unrank(k, grid.n) for k in range(start, stop)]
    states_array = np.array(states, dtype=np.int64)
    maslov_values, alexander_values = maslov_alexander_block(states_array, grid.O, grid.X)
    out_pairs, out_counts = differential_block(states_array, grid.O, grid.X)
    target_ranks = target_ranks_block(states_array, out_pairs, out_counts)
    maslov_list = maslov_values.tolist()
    alexander_list = alexander_values.tolist()
    counts_list = out_counts.tolist()
    records = []
    for offset in range(len(states)):
        count = counts_list[offset]
        records.append((
            start + offset,
            maslov_list[offset],
            alexander_list[offset],
            tuple(target_ranks[offset, :count].tolist()),
        ))
    return records


def _build_complexes(by_alexander: dict) -> dict:
    """Build {A: GradedComplex} from records grouped by Alexander grading. Generators are
    identified by lexicographic rank (an int); positions are assigned in group order, which the
    callers keep equal to global lexicographic (= rank) order, so this reproduces grid_complexes
    exactly. A differential column maps a generator's position to the positions of its targets one
    degree up."""
    complexes: dict = {}
    for a_grading, group in by_alexander.items():
        degree = {rank: d for rank, d, _ in group}
        position: dict = {}
        dims: dict = defaultdict(int)
        for rank, d, _ in group:
            position[rank] = dims[d]
            dims[d] += 1
        columns = {d: [None] * dims[d] for d in dims}
        for rank, d, targets in group:
            columns[d][position[rank]] = frozenset(
                position[target] for target in targets if degree.get(target) == d + 1
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


def grading_histogram(grid) -> dict:
    """Generator count per ``(Alexander, degree)`` grading over all n! states, gradings only.

    Builds no differentials and no matrices, so it is memory-safe at any n (it stores only the
    O(n^2) grading counts). The dimensions it returns are what set the reducer's dense-matrix
    sizes, so this is the input to ``dense_reduction_bytes`` -- a memory projection that can be
    computed for sizes far too large to actually generate or reduce. Walking n! states is itself
    cheap (gradings only), so it runs serially; the scheduler parallelizes the work that follows.
    """
    return _grading_slice((grid.O, grid.X, 0, math.factorial(grid.n)))
