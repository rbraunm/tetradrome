# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Adapter: express a knot's grid computation as scheduler jobs.

The histogram graph counts generators per grading over slices of the n! states and sums them;
its result prices everything after it. The whole-knot graph, built from that histogram, generates
in slices split by Alexander grading, merges each grading by its own job into its boundary maps,
takes each map's rank in its own job, and assembles the Poincare count from the ranks and the
histogram's dimensions. The tau graph does the same for the filtered complex in Maslov 0 and 1.
The reduction graph reduces complexes that already exist, one job per grading.

Every rank backend returns the identical answer by the agreement discipline, so a rank's compute
paths are a pure performance choice: a GPU path when a CUDA reducer is present, and a single-core
CPU path that is always there. Each path is priced by the cost model's packed peak. A forced
backend name ranks with that backend (this is how the agreement tests pin each tier), and
``auto`` lets the worker's environment decide, so a worker the scheduler placed on a GPU sees
the device and one placed on CPU does not.
"""
from __future__ import annotations

import math
from array import array
from collections import defaultdict

from ...algebra import (
    GradedComplex,
    available_f2_backends,
    best_available_backend,
    dense_block_bytes,
    dense_block_ops,
    f2_homology,
    f2_kernel_escape,
    f2_rank,
    predict_cost,
    predict_size,
    rank_backend,
    reduce_host_bytes,
)
from ...errors import TetradromeError
from ...scheduler import ComputePath, Job, JobGraph, Placement, Shard
from .generation import _generate_slice, _histogram_slice, _tau_slice
from .grid import GridDiagram

_REDUCE = "reduce"
_ASSEMBLE = "assemble"
_SLICE = "gen_slice"
_MERGE = "gen_merge"
_COLLECT = "gen_collect"
_RANK = "rank"
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
# granularity, never a worker count. A slice must run long against its dispatch: at grid number
# 10, 4096-state slices (0.15 s against 27 ms to dispatch) kept six to thirteen running; 32768
# (1.2 s) ran a six-knot batch in 139 s against 179 s.
_SLICE_STATES = 32768
_GEN_RECORD_BASE = 256
_GEN_RECORD_PER_N = 128
# What a forked worker makes private beyond its own working set: the inherited interpreter pages
# it touches. It grows with how long the worker lives, so it is sized from a loaded machine, where
# slice workers of a grid-number-10 knot peaked at 76 MB private against a working set of 9 MB,
# and tau rank and level workers ran up to 57 MB over their columns.
_FORK_BASE_BYTES = 80 * 2**20
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
    """One generation slice split by Alexander grading: for each grading asked for, its generators
    in this slice as ``(ranks, maslov, counts, targets)``, in rank order, with each generator's
    ``counts`` differential targets (lexicographic ranks) flattened in the same order."""
    import numpy as np

    o_markers, x_markers, start, stop, gradings = inputs
    generated = _generate_slice((o_markers, x_markers, start, stop))
    ranks = generated.start + np.arange(generated.maslov.shape[0], dtype=np.int64)
    valid = np.arange(generated.targetRanks.shape[1]) < generated.counts[:, None]
    parts = {}
    for alexander in gradings:
        members = generated.alexander == alexander
        parts[alexander] = (ranks[members], generated.maslov[members], generated.counts[members],
                            generated.targetRanks[members][valid[members]])
    return parts


def _index_array(values):
    # One memcpy into GradedComplex's 4-byte CSC typecode (its width is asserted at import there).
    import numpy as np

    buffer = array("i")
    buffer.frombytes(np.ascontiguousarray(values, dtype=np.int32).tobytes())
    return buffer


def _grading_merge_run(inputs, deps):
    """One Alexander grading's boundary maps, from that grading's share of every slice:
    ``{degree: (indices, indptr)}``, the CSC of the map out of each degree in ``inputs["degrees"]``.
    A generator's position within its degree is its place in rank order and each column's rows are
    sorted, so the maps are the ones ``grid_complexes`` builds, byte for byte. The differential
    keeps the Alexander grading and raises the degree by one, so every target is found in this
    grading, one degree up; the merge raises if one is not."""
    import numpy as np

    parts = sorted((part for part in deps.values() if part[0].shape[0]),
                   key=lambda part: int(part[0][0]))
    ranks, maslov, counts, targets = (
        np.concatenate([part[index] for part in parts]) if parts else np.empty(0, dtype=np.int64)
        for index in range(4))
    degree = -maslov
    source_degree = np.repeat(degree, counts)
    maps = {}
    for d in inputs["degrees"]:
        in_degree = degree == d
        rows = ranks[degree == d + 1]                        # ascending: a target's row is its index
        if (int(in_degree.sum()), rows.shape[0]) != (inputs["dims"][d], inputs["dims"][d + 1]):
            raise TetradromeError(
                f"generated {int(in_degree.sum())} and {rows.shape[0]} generators in degrees {d} "
                f"and {d + 1}; the grading histogram counted {inputs['dims'][d]} and "
                f"{inputs['dims'][d + 1]}")
        entries = targets[source_degree == d]
        positions = np.searchsorted(rows, entries)
        found = positions < rows.shape[0]
        found[found] = rows[positions[found]] == entries[found]
        if not found.all():
            raise TetradromeError(
                f"{int((~found).sum())} differential target(s) out of degree {d} are not "
                f"generators of degree {d + 1} in the same Alexander grading")
        column_counts = counts[in_degree]
        column_of = np.repeat(np.arange(column_counts.shape[0]), column_counts)
        positions = positions[np.lexsort((positions, column_of))]
        indptr = np.empty(column_counts.shape[0] + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(column_counts, out=indptr[1:])
        maps[d] = (_index_array(positions), _index_array(indptr))
    return maps


def _collect_run(inputs, deps):
    # {A: GradedComplex} from every grading's maps. A map with no entries is left out, as
    # GradedComplex stores it.
    by_grading: dict = defaultdict(dict)
    for shard, (indices, indptr) in deps.items():
        if len(indices):
            by_grading[shard.producer[1]][shard.key] = (indices, indptr)
    return {alexander: GradedComplex.from_csc(dims, by_grading[alexander])
            for alexander, dims in inputs["dims"].items()}


def _rank_run(inputs, deps):
    (boundary,) = deps.values()
    backend = best_available_backend() if inputs["backend"] == "auto" else inputs["backend"]
    return rank_backend(backend)(boundary, inputs["rows"])


def _rank_assemble_run(inputs, deps):
    """The Poincare count ``{(Maslov, Alexander): dimension}`` from every boundary map's rank and
    the chain dimensions the histogram counted."""
    rank = {key[1:]: value for key, value in deps.items()}      # key is (_RANK, alexander, degree)
    poincare = {}
    for (alexander, degree), dimension in inputs["dims"].items():
        betti = (dimension - rank.get((alexander, degree), 0)
                 - rank.get((alexander, degree - 1), 0))
        if betti < 0:
            raise RuntimeError(f"negative Betti number at Alexander grading {alexander}, degree "
                               f"{degree}: the boundary maps are not a complex.")
        if betti:
            poincare[(-degree, alexander)] = betti
    return poincare


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
    """``histogram_jobs`` as a JobGraph plus its merge key."""
    jobs, merge_key = histogram_jobs(grid, slice_states=slice_states)
    return JobGraph(jobs), merge_key


def histogram_jobs(grid, *, slice_states: int = _HISTOGRAM_SLICE_STATES) -> tuple:
    """Jobs counting generators per ``(Alexander, degree)`` over all n! states, plus the
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
                               ram_bytes=_FORK_BASE_BYTES + (stop - start) * state_bytes),),
            cost=float(stop - start),
        ))
        slice_keys.append(key)
    merge_key = (_HISTOGRAM_MERGE,)
    jobs.append(Job(
        key=merge_key, run=_histogram_merge_run, inputs={},
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=0),),
        dependencies=slice_keys,
    ))
    return jobs, merge_key


def _dims_by_grading(histogram: dict) -> dict:
    by_grading: dict = defaultdict(dict)
    for (alexander, degree), count in histogram.items():
        by_grading[alexander][degree] = count
    return dict(by_grading)


def generation_jobs(grid, histogram: dict, *, slice_states: int = _SLICE_STATES) -> tuple:
    """Generation jobs, as ``(jobs, maps)``: contiguous lexicographic slices of the n! permutation
    space, each split by Alexander grading, and one merge per grading that folds that grading's
    share of every slice into its boundary maps. ``maps`` is ``{alexander: (merge_key, degrees)}``:
    the merge's result is partitioned by degree, one shard per boundary map, for the gradings that
    have a map at all (two adjacent nonzero degrees).

    ``histogram`` is the grid's grading histogram. It says which gradings and degrees exist and
    how large each is, so every job is declared and priced before anything is generated, and no
    job ever holds more than one grading. The slice *count* falls out of ``n!`` at
    ``slice_states`` states each -- a math granularity, independent of the machine."""
    by_grading = _dims_by_grading(histogram)
    degrees_of = {alexander: tuple(d for d in sorted(dims) if dims.get(d + 1))
                  for alexander, dims in by_grading.items()}
    gradings = tuple(sorted(alexander for alexander, degrees in degrees_of.items() if degrees))
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
            inputs=(grid.O, grid.X, start, stop, gradings),
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=_FORK_BASE_BYTES + (stop - start) * record),),
            cost=float(stop - start), shards=frozenset(gradings),
        ))
        slice_keys.append(key)
    maps = {}
    for alexander in gradings:
        dims = by_grading[alexander]
        members = sum(dims.values())
        merge_key = (_MERGE, alexander)
        jobs.append(Job(
            key=merge_key, run=_grading_merge_run, kind="floer.generation_merge",
            inputs={"degrees": degrees_of[alexander], "dims": dims},
            # The merge holds this grading's records from every slice and builds its maps from
            # them; price the peak at both resident at once.
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=_FORK_BASE_BYTES + 2 * members * record),),
            dependencies={Shard(key, alexander) for key in slice_keys},
            cost=float(members), shards=frozenset(degrees_of[alexander]),
        ))
        maps[alexander] = (merge_key, degrees_of[alexander])
    return jobs, maps


def generation_graph(grid, histogram: dict, *, slice_states: int = _SLICE_STATES):
    """A JobGraph generating ``{A: GradedComplex}`` from ``grid``, plus the key of the job whose
    result equals the serial ``grid_complexes(grid)``."""
    jobs, maps = generation_jobs(grid, histogram, slice_states=slice_states)
    collect_key = (_COLLECT,)
    jobs.append(Job(
        key=collect_key, run=_collect_run, inputs={"dims": _dims_by_grading(histogram)},
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=0),),
        dependencies={Shard(merge_key, degree)
                      for merge_key, degrees in maps.values() for degree in degrees},
    ))
    return JobGraph(jobs), collect_key


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
    """``tau_jobs`` as a JobGraph plus its assemble key."""
    jobs, assemble_key = tau_jobs(grid, histogram, slice_states=slice_states)
    return JobGraph(jobs), assemble_key


def tau_jobs(grid, histogram: dict, *, slice_states: int = _TAU_SLICE_STATES) -> tuple:
    """The jobs for tau plus the assemble key, whose result is ``(Maslov-0 generator count,
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
                               ram_bytes=_FORK_BASE_BYTES + (stop - start) * record),),
            cost=float(stop - start),
        ))
        slice_keys.append(key)

    def serial(key, run, columns, dependencies, shards=None):
        return Job(
            key=key, run=run, inputs={}, kind=f"floer.{key[0]}",
            paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                               ram_bytes=_FORK_BASE_BYTES + columns * _TAU_COLUMN_BYTES),),
            dependencies=dependencies, cost=float(columns), shards=shards,
        )

    merge_key = (_TAU_MERGE,)
    shard = {name: Shard(merge_key, name) for name in ("d0", "d1", "alexander")}
    columns = count[0] + count[1]
    jobs.append(Job(
        key=merge_key, run=_tau_merge_run, inputs={}, kind="floer.tau_merge",
        paths=(ComputePath(Placement.CPU_PINNED, cores=1,
                           ram_bytes=_FORK_BASE_BYTES + 2 * (columns + count[-1]) * record),),
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
    return jobs, assemble_key


# The knot the benchmark runs: grid number 8, the largest whose three calculations take seconds.
# A real knot, because the unknot's complexes reduce far faster per operation than a knot's do.
_BENCHMARK_KNOT = "6_2"


def benchmark(run) -> None:
    """The Floer engine's representative calculations for ``tetradrome.calibrate``: the histogram,
    HFK-hat and tau graphs of one knot's grid, each run through ``run(graph, key)``."""
    grid = GridDiagram.from_knotinfo(_BENCHMARK_KNOT)
    histogram = run(*histogram_graph(grid))
    run(*whole_knot_graph(grid, histogram))
    run(*tau_graph(grid, histogram))


def whole_knot_graph(grid, histogram: dict, *, backend: str = "bitint"):
    """``whole_knot_jobs`` as a JobGraph plus its assemble key."""
    jobs, assemble_key = whole_knot_jobs(grid, histogram, backend=backend)
    return JobGraph(jobs), assemble_key


def whole_knot_jobs(grid, histogram: dict, *, backend: str = "bitint") -> tuple:
    """The jobs from a grid to its Poincare count, as ``(jobs, assemble_key)``: generation slices
    split by Alexander grading, a merge per grading partitioned by degree, the rank of each
    boundary map from its own shard, and an assembly that turns the ranks and the histogram's
    dimensions into the ``{(Maslov, Alexander): dimension}`` count.

    ``histogram`` is the grid's grading histogram (the ``histogram_graph`` result). Every job's
    footprint and cost come from it: a boundary map of ``cols`` columns into ``rows`` rows is
    priced by ``dense_block_bytes`` and ``dense_block_ops``, so the graph is fully priced before
    any generator exists."""
    jobs, maps = generation_jobs(grid, histogram)
    by_grading = _dims_by_grading(histogram)
    rank_keys = []
    for alexander, (merge_key, degrees) in maps.items():
        dims = by_grading[alexander]
        for degree in degrees:
            cols, rows = dims[degree], dims[degree + 1]
            key = (_RANK, alexander, degree)
            jobs.append(Job(
                key=key, run=_rank_run, inputs={"backend": backend, "rows": rows},
                kind=f"f2.reduce.{backend}",
                paths=_reduction_paths(dense_block_bytes(cols, rows), backend),
                cost=float(dense_block_ops(cols, rows)),
                dependencies={Shard(merge_key, degree)},
            ))
            rank_keys.append(key)
    assemble_key = (_ASSEMBLE,)
    jobs.append(Job(
        key=assemble_key, run=_rank_assemble_run, inputs={"dims": dict(histogram)},
        paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=0),),
        dependencies=rank_keys,
    ))
    return jobs, assemble_key
