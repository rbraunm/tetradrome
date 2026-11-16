# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Adapter: express a knot's grid computation as scheduler jobs.

Three graphs. The histogram graph counts generators per grading over slices of the n! states and
sums them; its result prices everything after it. The whole-knot graph, built from that
histogram, generates the complexes in slices, merges them partitioned by Alexander grading,
reduces each grading, and assembles the Poincare count. The reduction graph is the tail of the
whole-knot graph for complexes that already exist.

A grading's reduction is f2_homology on one complex, and every backend returns the identical
answer by the agreement discipline, so a reduction's compute paths are a pure performance
choice: a GPU path when a CUDA reducer is present, and a single-core CPU path that is always
there. The scheduler runs the fastest the machine can serve. Each path is priced by the cost
model's packed peak. The assembly fans the per-grading homologies into the bigraded Poincare
count, the same arithmetic the in-process reducer used.

This builds only the reduction-plus-assembly subgraph. The whole-knot builder prepends
generation and reuses it.

A reduction's run resolves its backend from its input: a forced backend name reduces with that
backend (this is how the agreement tests pin each tier), and ``auto`` lets the worker's
environment decide, so a worker the scheduler placed on a GPU sees the device and one placed on
CPU does not.
"""
from __future__ import annotations

import math
from array import array
from collections import defaultdict

from ...algebra import (
    available_f2_backends,
    f2_homology,
    f2_kernel_escape,
    f2_rank,
    grading_cost_ops,
    grading_peak_bytes,
    predict_cost,
    predict_size,
    reduce_host_bytes,
)
from ...errors import TetradromeError
from ...scheduler import ComputePath, Job, JobGraph, Placement, Shard
from .generation import _build_complexes, _generate_slice, _histogram_slice, _tau_slice

_REDUCE = "reduce"
_ASSEMBLE = "assemble"
_SLICE = "gen_slice"
_MERGE = "gen_merge"
_HISTOGRAM_SLICE = "hist_slice"
_HISTOGRAM_MERGE = "hist_merge"
_TAU_SLICE = "tau_slice"
_TAU_MERGE = "tau_merge"
_TAU_RANK = "tau_rank"
_TAU_LEVELS = "tau_levels"
_TAU_ASSEMBLE = "tau_assemble"

# Generation is not in the cost model (grading_histogram predicts reduction dims, not record
# bytes), so a slice's working set and the merge's peak are priced by a per-record estimate: a
# base plus a per-element term, tuned by the scheduler's over-budget warnings the same way every
# other footprint is. Slice count falls out of n! at this many states per slice, a math
# granularity, never a worker count.
_SLICE_STATES = 4096
_GEN_RECORD_BASE = 256
_GEN_RECORD_PER_N = 128
# What a forked slice worker dirties before it holds any state. The forkserver hands it numpy and
# the bound kernels (prewarm.py), so this is small: over the slices of a grid-number-10 knot the
# private peak was at most 12 MiB. A worker that binds the kernels itself needs about 64 MiB.
_SLICE_FORK_BASE_BYTES = 16 * 2**20
# A histogram slice holds its states (n int64 each), their two gradings, and the stacked pairs it
# counts; it builds no differential, so it takes far more states per job than a generation slice.
_HISTOGRAM_SLICE_STATES = 65536
_HISTOGRAM_STATE_BASE = 64
_HISTOGRAM_STATE_PER_N = 8
# A tau slice grades every state but builds the differential for Maslov 0 and 1 only, so it sits
# between the two: more states per job than a generation slice, priced at the generation record.
_TAU_SLICE_STATES = 16384
# The set-based reducer holds each column as a Python set of row indices; measured on a
# grid-number-10 knot at just over 1 KiB per column for both the rank and the level search.
_TAU_COLUMN_BYTES = 1152
_TAU_FORK_BASE_BYTES = 10 * 2**20


def _reduce_run(inputs, deps):
    return f2_homology(inputs["complex"], inputs["backend"])


def _assemble_run(inputs, deps):
    poincare: dict = defaultdict(int)
    for key, homology in deps.items():
        alexander = key[1]                       # key is (_REDUCE, alexander)
        for degree, dimension in homology.items():
            poincare[(-degree, alexander)] += dimension
    return {key: value for key, value in poincare.items() if value}


def _reduction_paths(peak: int, backend: str) -> tuple:
    # `peak` is the packed word-matrix peak (matrix + pivots). reduce_host_bytes turns it into the
    # host RAM the chosen backend's worker actually needs: a fork base plus a backend-specific
    # working set (the pure-Python tiers hold only pivots, the packed tiers the whole matrix).
    # The GPU VRAM is still priced at the packed peak; the dense uint8 matrix the GPU reducer
    # builds is several times larger, a separate correction pending a GPU benchmark.
    if backend == "packed-gpu":
        return (ComputePath(Placement.GPU, cores=1,
                            ram_bytes=reduce_host_bytes(peak, "packed-gpu"),
                            vram_bytes=max(peak, 1)),)
    if backend != "auto":
        return (ComputePath(Placement.CPU_PINNED, cores=1,
                            ram_bytes=reduce_host_bytes(peak, backend)),)
    available = {name: ok for name, ok, _ in available_f2_backends()}
    paths = []
    if available.get("packed-gpu"):
        paths.append(ComputePath(Placement.GPU, cores=1,
                                 ram_bytes=reduce_host_bytes(peak, "packed-gpu"),
                                 vram_bytes=max(peak, 1)))
    # The auto CPU fallback runs the fastest CPU tier, which route_backend resolves to bitint.
    paths.append(ComputePath(Placement.CPU_PINNED, cores=1,
                             ram_bytes=reduce_host_bytes(peak, "bitint")))
    return tuple(paths)


def reduction_jobs(complexes: dict, *, backend: str) -> tuple:
    """Reduction jobs (one per grading) plus the assembly that consumes them, as
    ``(jobs, assemble_key)``. The assembly depends on every reduction, so a failed reduction
    abandons the whole knot."""
    jobs = []
    reduce_keys = []
    for alexander, cx in complexes.items():
        key = (_REDUCE, alexander)
        jobs.append(Job(
            key=key, run=_reduce_run, kind=f"f2.reduce.{backend}",
            inputs={"complex": cx, "backend": backend},
            paths=_reduction_paths(predict_size(cx).packed_peak_bytes, backend),
            cost=predict_cost(cx),
        ))
        reduce_keys.append(key)
    assemble_key = (_ASSEMBLE,)
    jobs.append(Job(
        key=assemble_key, run=_assemble_run, inputs={},
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=0),),
        dependencies=reduce_keys,
    ))
    return jobs, assemble_key


def reduction_graph(complexes: dict, *, backend: str):
    """A JobGraph for reducing ``{A: GradedComplex}`` plus its assembly, and the assembly key
    whose result is the ``{(Maslov, Alexander): dimension}`` Poincare count."""
    jobs, assemble_key = reduction_jobs(complexes, backend=backend)
    return JobGraph(jobs), assemble_key


def _slice_run(inputs, deps):
    return _generate_slice(inputs)


def _merge_run(inputs, deps):
    # Assemble the slices (each a _GenerationSlice of arrays carrying its global start) into global
    # rank-indexed arrays -- rank is the array index, since the slices tile [0, n!) contiguously --
    # plus a CSR of the differential targets (a flat target-rank array with per-generator offsets),
    # then build the complexes. No per-generator Python on the fold; positions still land in
    # lexicographic (= rank) order, so the result equals grid_complexes.
    import numpy as np

    slices = list(deps.values())
    total = sum(int(s.maslov.shape[0]) for s in slices)
    maslov = np.empty(total, dtype=np.int64)
    alexander = np.empty(total, dtype=np.int64)
    counts = np.empty(total, dtype=np.int64)
    blocks = []
    for s in slices:
        size = int(s.maslov.shape[0])
        if size == 0:
            continue
        lo = s.start
        hi = lo + size
        maslov[lo:hi] = s.maslov
        alexander[lo:hi] = s.alexander
        counts[lo:hi] = s.counts
        keep = np.arange(s.targetRanks.shape[1]) < s.counts[:, None]
        blocks.append((lo, s.targetRanks[keep]))         # this slice's valid targets, in rank order
    blocks.sort(key=lambda block: block[0])              # by start, so all targets are in rank order
    targets_flat = (np.concatenate([targets for _, targets in blocks])
                    if blocks else np.empty(0, dtype=np.int64))
    offsets = np.empty(total + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return _build_complexes(maslov, alexander, targets_flat, offsets)


def _record_bytes(n: int) -> int:
    return _GEN_RECORD_BASE + _GEN_RECORD_PER_N * n


def _histogram_slice_run(inputs, deps):
    return _histogram_slice(inputs)


def _histogram_merge_run(inputs, deps):
    histogram: dict = defaultdict(int)
    for counts in deps.values():
        for grading, count in counts.items():
            histogram[grading] += count
    return dict(histogram)


def histogram_graph(grid, *, slice_states: int = _HISTOGRAM_SLICE_STATES):
    """A JobGraph counting generators per ``(Alexander, degree)`` over all n! states, plus the
    merge key whose result equals ``grading_histogram_reference(grid)``. Slices are contiguous
    ranges of the permutation space at ``slice_states`` states each, a math granularity like the
    generation slices; the merge sums their counts, so slice order cannot matter."""
    total = math.factorial(grid.n)
    slice_count = max(1, math.ceil(total / slice_states))
    state_bytes = _HISTOGRAM_STATE_BASE + _HISTOGRAM_STATE_PER_N * grid.n
    jobs = []
    slice_keys = []
    for w in range(slice_count):
        start = total * w // slice_count
        stop = total * (w + 1) // slice_count
        key = (_HISTOGRAM_SLICE, w)
        jobs.append(Job(
            key=key, run=_histogram_slice_run, kind="floer.histogram_slice",
            inputs=(grid.O, grid.X, start, stop),
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=_SLICE_FORK_BASE_BYTES + (stop - start) * state_bytes),),
            cost=float(stop - start),
        ))
        slice_keys.append(key)
    merge_key = (_HISTOGRAM_MERGE,)
    jobs.append(Job(
        key=merge_key, run=_histogram_merge_run, inputs={},
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=0),),
        dependencies=slice_keys,
    ))
    return JobGraph(jobs), merge_key


def generation_jobs(grid, *, slice_states: int = _SLICE_STATES, merge_shards=None) -> tuple:
    """Generation jobs (contiguous lexicographic slices of the n! permutation space) plus the
    merge that folds them into ``{A: GradedComplex}``, as ``(jobs, merge_key)``. The merge depends
    on every slice and folds them in slice-index order, so its result equals ``grid_complexes``.
    The slice *count* falls out of ``n!`` at ``slice_states`` states each -- a math granularity,
    independent of the machine. ``merge_shards``, when given, declares the merge's output
    partitioned over exactly those Alexander gradings, so each is stored and consumed independently
    (the merge already returns ``{A: GradedComplex}``, the shard shape)."""
    total = math.factorial(grid.n)
    slice_count = max(1, math.ceil(total / slice_states))
    record = _record_bytes(grid.n)
    jobs = []
    slice_keys = []
    for w in range(slice_count):
        start = total * w // slice_count
        stop = total * (w + 1) // slice_count
        key = (_SLICE, w)
        jobs.append(Job(
            key=key, run=_slice_run, kind="floer.generation_slice",
            inputs=(grid.O, grid.X, start, stop),
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=_SLICE_FORK_BASE_BYTES + (stop - start) * record),),
            cost=float(stop - start),
        ))
        slice_keys.append(key)
    merge_key = (_MERGE,)
    jobs.append(Job(
        key=merge_key, run=_merge_run, inputs={}, kind="floer.generation_merge",
        # The merge holds every slice's records and builds the complexes from them; price the peak
        # at both resident at once.
        paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                           ram_bytes=max(2 * total * record, 1)),),
        dependencies=slice_keys,
        cost=float(total),
        shards=merge_shards,
    ))
    return jobs, merge_key


def generation_graph(grid, *, slice_states: int = _SLICE_STATES):
    """A JobGraph generating ``{A: GradedComplex}`` from ``grid`` plus the merge key whose result
    equals the serial ``grid_complexes(grid)``."""
    jobs, merge_key = generation_jobs(grid, slice_states=slice_states)
    return JobGraph(jobs), merge_key


def _tau_slice_run(inputs, deps):
    return _tau_slice(inputs)


def _tau_merge_run(inputs, deps):
    # Fold the slices, in rank order, into the two boundary maps tau needs, each as CSC over
    # array('i'): d0 from the Maslov 0 generators to the Maslov -1 ones and d1 from Maslov 1 to
    # Maslov 0, rows and columns numbered by position in rank order -- the reference's matrices.
    import numpy as np

    slices = sorted(deps.values(), key=lambda piece: piece.start)

    def joined(field):
        return np.concatenate([getattr(piece, field) for piece in slices])

    def csc(targets, counts, row_ranks, rows):
        positions = np.searchsorted(row_ranks, targets)
        inside = positions < row_ranks.shape[0]
        if not inside.all() or not np.array_equal(row_ranks[positions], targets):
            raise TetradromeError(
                f"a filtered-differential target is not a Maslov {rows} generator; the filtered "
                f"differential must lower Maslov by exactly one.")
        indptr = np.zeros(counts.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return (array("i", positions.astype(np.int32).tobytes()),
                array("i", indptr.astype(np.int32).tobytes()))

    zero_ranks = joined("zeroRanks")
    return {
        "d0": csc(joined("zeroTargets"), joined("zeroCounts"), joined("minusRanks"), -1),
        "d1": csc(joined("oneTargets"), joined("oneCounts"), zero_ranks, 0),
        "alexander": array("i", joined("zeroAlexander").astype(np.int32).tobytes()),
    }


def _tau_rank_run(inputs, deps):
    (csc,) = deps.values()
    return f2_rank(csc)


def _csc_columns(csc):
    indices, indptr = csc
    return [frozenset(indices[indptr[k]:indptr[k + 1]]) for k in range(len(indptr) - 1)]


def _tau_levels_run(inputs, deps):
    # The first Alexander level whose cycles (the kernel of d0 on that level and below) are not
    # all boundaries (the image of d1), with rank d1.
    merge_key = (_TAU_MERGE,)
    return f2_kernel_escape(_csc_columns(deps[Shard(merge_key, "d0")]),
                            deps[Shard(merge_key, "alexander")],
                            _csc_columns(deps[Shard(merge_key, "d1")]))


def _tau_assemble_run(inputs, deps):
    level, rank_d1 = deps[(_TAU_LEVELS,)]
    return len(deps[Shard((_TAU_MERGE,), "alexander")]), deps[(_TAU_RANK,)], rank_d1, level


def tau_graph(grid, histogram: dict, *, slice_states: int = _TAU_SLICE_STATES):
    """A JobGraph for tau plus the assemble key, whose result is ``(Maslov-0 generator count,
    rank d0, rank d1, level)``: the two ranks that fix dim H_0 of the filtered complex, and the
    first Alexander level carrying the surviving class (None if no level does). Slices grade
    the n! states and build the filtered differential of the Maslov 0 and 1 generators; a merge
    folds them into the two boundary maps; rank d0 and the level search then run as two jobs,
    the level search yielding rank d1 along the way.

    ``histogram`` is the grid's grading histogram; its Maslov -1, 0 and 1 generator counts price
    the merge and the linear algebra before any matrix exists."""
    total = math.factorial(grid.n)
    slice_count = max(1, math.ceil(total / slice_states))
    record = _record_bytes(grid.n)
    count = {maslov: sum(generators for (_, degree), generators in histogram.items()
                         if degree == -maslov) for maslov in (-1, 0, 1)}
    jobs = []
    slice_keys = []
    for w in range(slice_count):
        start = total * w // slice_count
        stop = total * (w + 1) // slice_count
        key = (_TAU_SLICE, w)
        jobs.append(Job(
            key=key, run=_tau_slice_run, kind="floer.tau_slice",
            inputs=(grid.O, grid.X, start, stop),
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=_SLICE_FORK_BASE_BYTES + (stop - start) * record),),
            cost=float(stop - start),
        ))
        slice_keys.append(key)

    def serial(key, run, columns, dependencies, shards=None):
        return Job(
            key=key, run=run, inputs={}, kind=f"floer.{key[0]}",
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=_TAU_FORK_BASE_BYTES + columns * _TAU_COLUMN_BYTES),),
            dependencies=dependencies, cost=float(columns), shards=shards,
        )

    merge_key = (_TAU_MERGE,)
    shard = {name: Shard(merge_key, name) for name in ("d0", "d1", "alexander")}
    columns = count[0] + count[1]
    jobs.append(Job(
        key=merge_key, run=_tau_merge_run, inputs={}, kind="floer.tau_merge",
        paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                           ram_bytes=_SLICE_FORK_BASE_BYTES + 2 * (columns + count[-1]) * record),),
        dependencies=slice_keys, cost=float(columns), shards=frozenset(shard),
    ))
    jobs.append(serial((_TAU_RANK,), _tau_rank_run, count[0], {shard["d0"]}))
    jobs.append(serial((_TAU_LEVELS,), _tau_levels_run, 2 * (2 * count[0] + count[1]),
                       {shard["d0"], shard["d1"], shard["alexander"]}))
    assemble_key = (_TAU_ASSEMBLE,)
    jobs.append(Job(
        key=assemble_key, run=_tau_assemble_run, inputs={},
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=0),),
        dependencies={shard["alexander"], (_TAU_RANK,), (_TAU_LEVELS,)},
    ))
    return JobGraph(jobs), assemble_key


def _wired_reduce_run(inputs, deps):
    # The single dep is this grading's complex -- the merge shard addressed by Shard(merge_key, A).
    (complex_for_grading,) = deps.values()
    return f2_homology(complex_for_grading, inputs["backend"])


def whole_knot_graph(grid, histogram: dict, *, backend: str = "bitint"):
    """One graph from a grid to its Poincare count: generation slices feed a merge partitioned by
    Alexander grading, each grading is reduced from its own shard (reading only that grading's
    complex, never the whole output), and an assembly folds the per-grading homologies into the
    ``{(Maslov, Alexander): dimension}`` count. Returned as ``(JobGraph, assemble_key)``.

    ``histogram`` is the grid's grading histogram (the ``histogram_graph`` result). The shard set
    and the per-grading reduction footprints come from it: ``grading_peak_bytes`` and
    ``grading_cost_ops`` over a grading's ``{degree: dim}`` equal the materialized complex's
    predicted peak and cost, so the graph is fully priced before any complex exists. The merge
    returns exactly those gradings, so the partition matches."""
    by_grading: dict = defaultdict(dict)
    for (alexander, degree), count in histogram.items():
        by_grading[alexander][degree] = count
    gradings = frozenset(by_grading)
    jobs, merge_key = generation_jobs(grid, merge_shards=gradings)
    reduce_keys = []
    for alexander, degree_dims in by_grading.items():
        key = (_REDUCE, alexander)
        jobs.append(Job(
            key=key, run=_wired_reduce_run, inputs={"backend": backend},
            kind=f"f2.reduce.{backend}",
            paths=_reduction_paths(grading_peak_bytes(degree_dims), backend),
            cost=grading_cost_ops(degree_dims),
            dependencies={Shard(merge_key, alexander)},
        ))
        reduce_keys.append(key)
    assemble_key = (_ASSEMBLE,)
    jobs.append(Job(
        key=assemble_key, run=_assemble_run, inputs={},
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=0),),
        dependencies=reduce_keys,
    ))
    return JobGraph(jobs), assemble_key
