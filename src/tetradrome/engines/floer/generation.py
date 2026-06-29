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
from array import array
from collections import defaultdict, namedtuple

from ...algebra import GradedComplex
from ...errors import BackendUnavailable
from .differential import differential
from .differential_jit import (
    HAVE_NUMPY,
    differential_block,
    target_ranks_block,
)
from .gradings import alexander, maslov
from .gradings_jit import maslov_alexander_block
from .grid import GridDiagram

# A generation slice's result: columnar arrays indexed by slice offset (so generator
# ``start + offset``), carrying the slice's global ``start`` so the merge places each block at its
# rank range. Arrays rather than a list of per-generator records, so the parent's result transport
# is a near-contiguous copy instead of one tuple per generator. The scheduled path requires numpy
# (the kernels do); grid_complexes, _grading_slice, and grading_histogram stay pure Python --
# grid_complexes as the canonical, auditable oracle and the numpy-free escape hatch, the grading
# walk because it must stay memory-safe at sizes too large to materialize.
_GenerationSlice = namedtuple("_GenerationSlice", "start maslov alexander targetRanks counts")


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
    """One contiguous lexicographic slice of the generation step as columnar arrays:
    ``_GenerationSlice(start, maslov, alexander, targetRanks, counts)`` indexed by slice offset (so
    generator ``start + offset``). The numba kernels compute the block gradings, the differential's
    surviving transposition pairs, and those targets' lexicographic ranks over the whole slice, and
    the result is those arrays -- no per-generator Python on the output side, so the parent's result
    transport is a near-contiguous copy rather than one tuple per generator. Requires numpy (the
    kernels do); for a pure-Python complex without numpy use ``grid_complexes``. Folded by
    ``_merge_run``, which ``test_generation_graph_matches_serial`` pins against ``grid_complexes``."""
    o_markers, x_markers, start, stop = args
    if not HAVE_NUMPY:
        raise BackendUnavailable(
            "the scheduled generation path requires numpy (pip install the 'jit' or 'accel' "
            "extra); for a pure-Python complex without numpy use grid_complexes."
        )
    import numpy as np

    grid = GridDiagram(o_markers, x_markers)
    if start >= stop:
        empty = np.empty(0, dtype=np.int64)
        return _GenerationSlice(start, empty, empty, np.empty((0, 0), dtype=np.int64), empty)
    states = np.array([_unrank(k, grid.n) for k in range(start, stop)], dtype=np.int64)
    maslov_values, alexander_values = maslov_alexander_block(states, grid.O, grid.X)
    out_pairs, out_counts = differential_block(states, grid.O, grid.X)
    target_ranks = target_ranks_block(states, out_pairs, out_counts)
    return _GenerationSlice(start, maslov_values, alexander_values, target_ranks, out_counts)


def _build_complexes(maslov_values, alexander_values, targets_flat, offsets):
    """Build ``{A: GradedComplex}`` from rank-indexed global arrays: generator ``r`` has gradings
    ``maslov_values[r]`` / ``alexander_values[r]`` and differential targets
    ``targets_flat[offsets[r]:offsets[r + 1]]`` (target ranks). Generators are identified by
    lexicographic rank = array index; positions are assigned per (Alexander, degree) in rank order,
    reproducing grid_complexes. The differential is degree-raising and Alexander-preserving, so the
    surviving targets and their positions are resolved in bulk over the whole flat target array, and
    each grading's CSC column buffers are then sliced out per (Alexander, degree) block with a single
    integer comparison -- no per-generator Python and no intermediate frozensets."""
    import numpy as np

    total = maslov_values.shape[0]
    if total == 0:
        return {}

    def as_index_array(values):
        # Copy a numpy int array into an array('i') buffer in one memcpy (no per-element Python);
        # 'i' is GradedComplex's 4-byte CSC typecode (its width is asserted at import there).
        buf = array("i")
        buf.frombytes(np.ascontiguousarray(values, dtype=np.int32).tobytes())
        return buf

    degree = -maslov_values

    # Pass 1: each generator's position within its (grading, degree) block, in rank order. One global
    # array, filled per grading -- the blocks are disjoint, so every entry is written exactly once.
    position = np.empty(total, dtype=np.int64)
    gradings = np.unique(alexander_values)
    members_of = {}
    dims_of = {}
    for a_value in gradings:
        members = np.nonzero(alexander_values == a_value)[0]      # ranks in grading, ascending
        members_of[int(a_value)] = members
        member_degrees = degree[members]
        dims: dict = {}
        for d in np.unique(member_degrees):
            in_degree = members[member_degrees == d]
            position[in_degree] = np.arange(in_degree.shape[0], dtype=np.int64)
            dims[int(d)] = int(in_degree.shape[0])
        dims_of[int(a_value)] = dims

    # Bulk: keep only targets one degree above their source (the rest are not in the column), then
    # resolve every surviving target to its position -- all vectorized over the flat target array.
    counts = np.diff(offsets)
    source_degree = np.repeat(degree, counts)                    # the source's degree, per target
    raised = degree[targets_flat] == source_degree + 1
    raised_positions = position[targets_flat[raised]]
    raised_cumulative = np.empty(targets_flat.shape[0] + 1, dtype=np.int64)
    raised_cumulative[0] = 0
    np.cumsum(raised, out=raised_cumulative[1:])
    raised_start = raised_cumulative[offsets]                    # per-generator span in raised_positions

    # Label each surviving target with its source generator and that source's (Alexander, degree)
    # block, so a whole block's column buffer is one boolean select over the flat position array.
    gen_raised_count = np.diff(raised_start)                     # raised targets per generator
    source_of_raised = np.repeat(np.arange(total, dtype=np.int64), gen_raised_count)
    degree_min = int(degree.min())
    degree_span = int(degree.max()) - degree_min + 1
    alexander_min = int(alexander_values.min())
    block_key = (alexander_values.astype(np.int64) - alexander_min) * degree_span + (
        degree.astype(np.int64) - degree_min
    )
    block_key_of_raised = block_key[source_of_raised]

    # Pass 2: slice each grading's CSC out of the resolved arrays. Within a block, generators are in
    # rank order = position order, and a block's raised entries sit consecutively per source in
    # raised_positions, so the boolean select yields columns in position order while the per-column
    # counts give indptr -- the two agree column-for-column, empty columns included.
    complexes: dict = {}
    for a_value in gradings:
        a_grading = int(a_value)
        dims = dims_of[a_grading]
        members = members_of[a_grading]
        member_degrees = degree[members]
        csc: dict = {}
        for d in dims:
            block = (a_grading - alexander_min) * degree_span + (d - degree_min)
            indices = raised_positions[block_key_of_raised == block]
            if indices.shape[0] == 0:
                continue  # a zero map: from_csc omits it and differential returns it empty
            in_degree = members[member_degrees == d]              # ascending = position order
            column_counts = gen_raised_count[in_degree]
            # Sort row indices within each column so the CSC is byte-identical to the column
            # constructor (grid_complexes sorts each column) -- the bit-for-bit agreement contract.
            column_of = np.repeat(np.arange(in_degree.shape[0]), column_counts)
            indices = indices[np.lexsort((indices, column_of))]
            indptr = np.empty(in_degree.shape[0] + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(column_counts, out=indptr[1:])
            csc[d] = (as_index_array(indices), as_index_array(indptr))
        complexes[a_grading] = GradedComplex.from_csc(dims, csc)
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
