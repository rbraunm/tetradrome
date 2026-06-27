# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Adapter: express a knot's F2 reduction as scheduler jobs.

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
from collections import defaultdict

from ...algebra import (
    available_f2_backends,
    f2_homology,
    grading_cost_ops,
    grading_peak_bytes,
    predict_cost,
    predict_size,
)
from ...scheduler import ComputePath, Job, JobGraph, Placement, Shard
from .generation import _build_complexes, _generate_slice, grading_histogram

_REDUCE = "reduce"
_ASSEMBLE = "assemble"
_SLICE = "gen_slice"
_MERGE = "gen_merge"

# Generation is not in the cost model (grading_histogram predicts reduction dims, not record
# bytes), so a slice's working set and the merge's peak are priced by a per-record estimate: a
# base plus a per-element term, tuned by the scheduler's over-budget warnings the same way every
# other footprint is. Slice count falls out of n! at this many states per slice, a math
# granularity, never a worker count.
_SLICE_STATES = 4096
_GEN_RECORD_BASE = 256
_GEN_RECORD_PER_N = 128


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
    if backend == "packed-gpu":
        return (ComputePath(Placement.GPU, cores=1, ram_bytes=peak, vram_bytes=max(peak, 1)),)
    if backend != "auto":
        return (ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=peak),)
    available = {name: ok for name, ok, _ in available_f2_backends()}
    paths = []
    if available.get("packed-gpu"):
        paths.append(ComputePath(Placement.GPU, cores=1, ram_bytes=peak, vram_bytes=max(peak, 1)))
    paths.append(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=peak))
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
            key=key, run=_reduce_run,
            inputs={"complex": cx, "backend": backend},
            paths=_reduction_paths(predict_size(cx).packed_peak_bytes, backend),
            cost=predict_cost(cx),
        ))
        reduce_keys.append(key)
    assemble_key = (_ASSEMBLE,)
    jobs.append(Job(
        key=assemble_key, run=_assemble_run, inputs={},
        paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=0),),
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
            key=key, run=_slice_run,
            inputs=(grid.O, grid.X, start, stop),
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=max((stop - start) * record, 1)),),
            cost=float(stop - start),
        ))
        slice_keys.append(key)
    merge_key = (_MERGE,)
    jobs.append(Job(
        key=merge_key, run=_merge_run, inputs={},
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


def _wired_reduce_run(inputs, deps):
    # The single dep is this grading's complex -- the merge shard addressed by Shard(merge_key, A).
    (complex_for_grading,) = deps.values()
    return f2_homology(complex_for_grading, inputs["backend"])


def whole_knot_graph(grid, *, backend: str = "bitint"):
    """One graph from a grid to its Poincare count: generation slices feed a merge partitioned by
    Alexander grading, each grading is reduced from its own shard (reading only that grading's
    complex, never the whole output), and an assembly folds the per-grading homologies into the
    ``{(Maslov, Alexander): dimension}`` count. Returned as ``(JobGraph, assemble_key)``.

    The shard set and the per-grading reduction footprints come from the build-time grading
    histogram: ``grading_peak_bytes`` and ``grading_cost_ops`` over a grading's ``{degree: dim}``
    equal the materialized complex's predicted peak and cost, so the graph is fully priced before
    any complex exists. The merge returns exactly those gradings, so the partition matches."""
    by_grading: dict = defaultdict(dict)
    for (alexander, degree), count in grading_histogram(grid).items():
        by_grading[alexander][degree] = count
    gradings = frozenset(by_grading)
    jobs, merge_key = generation_jobs(grid, merge_shards=gradings)
    reduce_keys = []
    for alexander, degree_dims in by_grading.items():
        key = (_REDUCE, alexander)
        jobs.append(Job(
            key=key, run=_wired_reduce_run, inputs={"backend": backend},
            paths=_reduction_paths(grading_peak_bytes(degree_dims), backend),
            cost=grading_cost_ops(degree_dims),
            dependencies={Shard(merge_key, alexander)},
        ))
        reduce_keys.append(key)
    assemble_key = (_ASSEMBLE,)
    jobs.append(Job(
        key=assemble_key, run=_assemble_run, inputs={},
        paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=0),),
        dependencies=reduce_keys,
    ))
    return JobGraph(jobs), assemble_key
