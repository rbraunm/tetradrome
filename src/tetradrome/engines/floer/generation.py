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

The grading histogram walks the same permutation space but keeps only the per-grading counts
(no differential, no matrices), so it predicts the reducer's dense-matrix dimensions at sizes
far too large to build -- the input to the reduction-memory model. ``_histogram_slice`` counts one
slice with the block grading kernel and is what the scheduler fans out;
``grading_histogram_reference`` is the serial pure-Python walk it must equal.
"""
from __future__ import annotations

import math
from collections import defaultdict, namedtuple

from ...algebra import GradedComplex
from ...errors import BackendUnavailable
from .differential import differential
from .differential_jit import (
    HAVE_NUMPY,
    differential_block,
    filtered_differential_block,
    target_ranks_block,
    unrank_block,
)
from .gradings import alexander, maslov
from .gradings_jit import maslov_alexander_block
from .grid import GridDiagram

# A generation slice's result: columnar arrays indexed by slice offset (so generator
# ``start + offset``), carrying the slice's global ``start`` so the merge places each block at its
# rank range. Arrays rather than a list of per-generator records, so the parent's result transport
# is a near-contiguous copy instead of one tuple per generator. The scheduled path requires numpy
# (the kernels do); grid_complexes, _grading_slice, and grading_histogram_reference stay pure
# Python: the canonical, auditable oracles and the numpy-free escape hatch.
_GenerationSlice = namedtuple("_GenerationSlice", "start maslov alexander targetRanks counts")


# A tau slice's result: what the Alexander-filtered complex needs from one slice. Generators are
# named by lexicographic rank. For Maslov 0 and Maslov 1 it carries their ranks and their filtered
# differential's target ranks, flattened with a per-generator count; for Maslov 0 also the
# Alexander gradings; for Maslov -1 only the ranks, which index the rows Maslov 0 maps into.
_TauSlice = namedtuple(
    "_TauSlice",
    "start minusRanks zeroRanks zeroAlexander zeroTargets zeroCounts oneRanks oneTargets oneCounts")


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
    kernels do); for a pure-Python complex without numpy use ``grid_complexes``. Split by grading
    and folded per grading in ``scheduling``, which ``test_generation_graph_matches_serial`` pins
    against ``grid_complexes``."""
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
    states = unrank_block(start, stop, grid.n)
    maslov_values, alexander_values = maslov_alexander_block(states, grid.O, grid.X)
    out_pairs, out_counts = differential_block(states, grid.O, grid.X)
    target_ranks = target_ranks_block(states, out_pairs, out_counts)
    return _GenerationSlice(start, maslov_values, alexander_values, target_ranks, out_counts)


def _tau_slice(args):
    """One contiguous lexicographic slice of what tau needs, as a ``_TauSlice``: the block kernels
    grade every generator of the slice, then compute the filtered differential's targets for the
    Maslov 0 and Maslov 1 generators only."""
    o_markers, x_markers, start, stop = args
    if not HAVE_NUMPY:
        raise BackendUnavailable(
            "the scheduled tau path requires numpy (pip install the 'jit' or 'accel' extra); for "
            "a pure-Python tau without numpy use tau_reference."
        )
    import numpy as np

    grid = GridDiagram(o_markers, x_markers)
    states = unrank_block(start, stop, grid.n)
    maslov_values, alexander_values = maslov_alexander_block(states, grid.O, grid.X)
    ranks = np.arange(start, stop, dtype=np.int64)

    def columns(degree):
        chosen = np.nonzero(maslov_values == degree)[0]
        block = states[chosen]
        pairs, counts = filtered_differential_block(block, grid.O)
        targets = target_ranks_block(block, pairs, counts)
        valid = np.arange(targets.shape[1]) < counts[:, None]
        return chosen, targets[valid], counts

    zero, zero_targets, zero_counts = columns(0)
    one, one_targets, one_counts = columns(1)
    return _TauSlice(start, ranks[maslov_values == -1],
                     ranks[zero], alexander_values[zero], zero_targets, zero_counts,
                     ranks[one], one_targets, one_counts)


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


def _histogram_slice(args):
    """Generator count per (Alexander, degree=-Maslov) over one contiguous slice, by the block
    grading kernel: the scheduled form of ``_grading_slice``. Holds only its own slice's states,
    so it is memory-safe at any n."""
    o_markers, x_markers, start, stop = args
    if not HAVE_NUMPY:
        raise BackendUnavailable(
            "the scheduled grading histogram requires numpy (pip install the 'jit' or 'accel' "
            "extra); for a pure-Python count without numpy use grading_histogram_reference."
        )
    import numpy as np

    if start >= stop:
        return {}
    grid = GridDiagram(o_markers, x_markers)
    states = unrank_block(start, stop, grid.n)
    maslov_values, alexander_values = maslov_alexander_block(states, grid.O, grid.X)
    gradings, counts = np.unique(
        np.stack((alexander_values, -maslov_values), axis=1), axis=0, return_counts=True)
    return {(int(a_grading), int(degree)): int(count)
            for (a_grading, degree), count in zip(gradings, counts)}


def grading_histogram_reference(grid) -> dict:
    """Generator count per ``(Alexander, degree)`` grading over all n! states: the serial
    pure-Python walk, kept as the oracle the scheduled histogram is pinned against."""
    return _grading_slice((grid.O, grid.X, 0, math.factorial(grid.n)))
