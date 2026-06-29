# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Complex-size prediction and backend routing (engine Phase 5).

`predict_size` reads a built complex and reports exact structural facts -- per-degree
dimensions, total generators, the largest boundary map, and the peak bytes of the packed
representation during reduction (the VRAM-relevant figure: the full matrix plus its pivots
for the heaviest grading). `route_backend` turns that into a decision: send a complex to
the GPU only when it both fits the measured VRAM budget (from `gpu.gpu_config`) and is
large enough to be worth the device overhead, otherwise to the fastest CPU tier; and, when
a RAM budget is supplied and the prediction exceeds it, fail loudly rather than attempt a
reduction that would OOM.

The GPU size threshold is a heuristic crossover, not a law of nature -- it defaults
conservatively and is meant to be calibrated with the benchmark on real hardware (the
small dense regime favours the CPU tiers; see design section 6). It is an explicit knob,
never a hidden default that silently sends tiny work to the wrong place.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping

from . import gpu

_DEFAULT_GPU_MIN_BYTES = 32 * 2**20      # below this the CPU tiers win; calibrate per box


@dataclasses.dataclass(frozen=True)
class ComplexSize:
    degrees: tuple[int, ...]
    dims: dict[int, int]                 # dim C^n
    total_generators: int
    largest_map: tuple[int, int, int]    # (degree n, cols = dim C^n, rows = dim C^{n+1})
    packed_peak_bytes: int               # full matrix + pivots, heaviest grading (uint64)


@dataclasses.dataclass(frozen=True)
class Routing:
    backend: str
    reason: str
    predicted_bytes: int
    fits_gpu: bool


def dense_block_bytes(cols: int, rows: int) -> int:
    """Bytes the packed F2 reducer holds for one boundary block of ``cols`` columns into
    ``rows`` rows: the packed matrix plus the pivot columns it accumulates (at most
    ``min(cols, rows)``), each column ``ceil(rows / 64)`` uint64 words. This matches what
    ``reduce_f2_packed.f2_rank_words`` actually allocates. Zero when either side is empty --
    a map into or out of a zero-dimensional space has no matrix to reduce.
    """
    if cols == 0 or rows == 0:
        return 0
    nwords = (rows + 63) // 64
    return (cols + min(cols, rows)) * nwords * 8


def dense_block_ops(cols: int, rows: int) -> int:
    """Word-XOR operations to F2-reduce one boundary block of ``cols`` columns into ``rows``
    rows: each of ``cols`` columns is eliminated against up to ``min(cols, rows)`` accumulated
    pivots, and each elimination touches ``ceil(rows / 64)`` uint64 words. Zero when either
    side is empty. This is the work analog of ``dense_block_bytes`` and is backend-independent;
    wall time is this count times a per-backend per-op cost the scheduler calibrates from
    telemetry.
    """
    if cols == 0 or rows == 0:
        return 0
    nwords = (rows + 63) // 64
    return cols * min(cols, rows) * nwords


def grading_peak_bytes(degree_dims: Mapping[int, int]) -> int:
    """Peak packed-reduction bytes for a single grading whose chain dimensions are
    ``degree_dims`` (``{degree: dim}``). The reducer holds one boundary block
    d^n: C^n -> C^(n+1) at a time, so the peak is the largest block across degrees.
    """
    peak = 0
    for n, cols in degree_dims.items():
        block = dense_block_bytes(cols, degree_dims.get(n + 1, 0))
        if block > peak:
            peak = block
    return peak


# --- backend-aware reduce-worker host memory (calibrated on CT 250, A8) ----------------------
#
# A reduce worker is a forked process reducing one grading; its peak resident memory is a
# per-backend fork base plus the reducer's working set. The single packed-peak figure the
# scheduler used to price every reduction was the working set of only the packed CPU tiers, and
# it omitted the fork base entirely -- so it over-provisioned the pure-Python default (which never
# builds the matrix) and under-provisioned every backend's small gradings (almost all fork base).
# Since the array('i') migration the CSC input a worker holds is ~25x smaller than the old
# frozensets and tiny beside the working set on the heavy gradings, so it is absorbed into the
# fork base rather than priced separately (pricing also runs from a dimension histogram, before a
# complex's nonzero count exists, so the footprint cannot be an exact separate term there).

# Per-worker fork base: the interpreter plus the modules the reducer dirties. The pure-Python
# tiers import only the stdlib; the packed tiers import numpy, and jit additionally numba, whose
# compiled code dominates a grading with little matrix. A8's per-reduce-job private memory settled
# near 6 MiB under bitint and near 20 MiB under jit for near-empty gradings; rounded up for margin.
_REDUCE_FORK_BASE_BYTES = {
    "reference": 10 * 2**20,
    "bitint": 10 * 2**20,
    "jit": 40 * 2**20,
    "packed-cpu": 40 * 2**20,
    "packed-gpu": 40 * 2**20,
}
_DEFAULT_REDUCE_FORK_BASE_BYTES = 40 * 2**20

# Working set as a fraction of the packed peak. The packed CPU tiers materialize the full word
# matrix plus pivots, so the packed peak is exactly their working set (1.0). The pure-Python tiers
# never build the matrix, only the pivot bit-vectors -- ~0.2x of the packed peak by measurement --
# so pricing them at the full packed peak (as the old model did) chronically over-provisions; 0.5
# keeps a safety margin over the measured 0.2x without paying for the matrix they never build.
_REDUCE_WORKING_FRACTION = {
    "reference": 0.5,
    "bitint": 0.5,
    "jit": 1.0,
    "packed-cpu": 1.0,
    "packed-gpu": 1.0,
}
_DEFAULT_REDUCE_WORKING_FRACTION = 1.0


def reduce_host_bytes(packed_peak: int, backend: str) -> int:
    """Host RAM a forked reduce worker needs for ``backend`` on a grading whose packed peak (word
    matrix plus pivots) is ``packed_peak``: a per-backend fork base plus the reducer's working set.

    Backend-aware because the reducers differ: the pure-Python ``reference``/``bitint`` tiers hold
    only the pivots (a fraction of the packed peak), the packed tiers hold the whole word matrix,
    and every backend carries a fork base the old single-figure model omitted. The base also
    absorbs the now-small CSC input footprint. Calibrated from the CT 250 A8 run; an unknown
    backend falls back to the heaviest assumption (numpy fork base, full packed peak), never the
    cheapest.
    """
    base = _REDUCE_FORK_BASE_BYTES.get(backend, _DEFAULT_REDUCE_FORK_BASE_BYTES)
    fraction = _REDUCE_WORKING_FRACTION.get(backend, _DEFAULT_REDUCE_WORKING_FRACTION)
    return base + int(packed_peak * fraction)


def predict_size(cx) -> ComplexSize:
    """Exact size facts for a GradedComplex, including the packed reduction peak."""
    degrees = tuple(cx.degrees())
    dims = {n: cx.dim(n) for n in degrees}
    total = sum(dims.values())
    peak = 0
    largest = (0, 0, 0)
    for n in degrees:
        block = dense_block_bytes(dims[n], cx.dim(n + 1))
        if block > peak:
            peak, largest = block, (n, dims[n], cx.dim(n + 1))
    return ComplexSize(degrees, dims, total, largest, peak)


def predict_cost(cx) -> int:
    """Total word-XOR operations to reduce a GradedComplex over F2: the sum over degrees of the
    work on each boundary block d^n: C^n -> C^(n+1). An op-count, not a time. The scheduler
    turns it into a predicted time with a per-backend constant it calibrates from observed
    runtimes, and uses that to decide whether a job is substantial enough to warrant its own
    process.
    """
    return sum(dense_block_ops(cx.dim(n), cx.dim(n + 1)) for n in cx.degrees())


def grading_cost_ops(degree_dims: Mapping[int, int]) -> int:
    """Total word-XOR operations to reduce a single grading whose chain dimensions are
    ``degree_dims`` (``{degree: dim}``), summed over its boundary blocks. The op-count analog of
    ``predict_cost`` taken from a dimension histogram rather than a built complex, so a reduction
    can be priced before its complex exists. For a grading A it equals ``predict_cost`` of that
    grading's complex, since both sum ``dense_block_ops(dim(n), dim(n + 1))`` over the same degrees.
    """
    return sum(dense_block_ops(cols, degree_dims.get(n + 1, 0)) for n, cols in degree_dims.items())


def route_backend(
    cx_or_size,
    available,
    *,
    gpu_cfg: gpu.GpuConfig | None = None,
    prefer_gpu: bool = True,
    gpu_min_bytes: int = _DEFAULT_GPU_MIN_BYTES,
    ram_budget_bytes: int | None = None,
) -> Routing:
    """Choose an F2 backend for one complex. `available` is the list from
    `tiers.available_f2_backends()`. GPU only when it fits the VRAM budget and clears the
    size threshold; CPU (bitint) otherwise; raise if a RAM budget is given and exceeded."""
    size = cx_or_size if isinstance(cx_or_size, ComplexSize) else predict_size(cx_or_size)
    avail = {name: ok for name, ok, _ in available}
    if gpu_cfg is None:
        gpu_cfg = gpu.gpu_config()
    pb = size.packed_peak_bytes

    if ram_budget_bytes is not None and pb > ram_budget_bytes:
        raise MemoryError(
            f"predicted packed peak {pb} bytes exceeds RAM budget {ram_budget_bytes} -- "
            "infeasible for this backend; reduce the problem or raise the budget."
        )

    vram_bytes = (gpu_cfg.vram_budget_mib or 0) * 2**20
    fits_gpu = bool(gpu_cfg.enabled and gpu_cfg.vram_budget_mib and pb <= vram_bytes)
    worth_gpu = pb >= gpu_min_bytes

    if prefer_gpu and avail.get("packed-gpu") and fits_gpu and worth_gpu:
        return Routing("packed-gpu",
                       f"fits VRAM budget ({pb >> 20} MiB <= {gpu_cfg.vram_budget_mib} MiB) "
                       f"and above the {gpu_min_bytes >> 20} MiB GPU threshold", pb, True)

    cpu = "bitint" if avail.get("bitint") else "reference"
    if gpu_cfg.enabled and not worth_gpu:
        reason = f"below the {gpu_min_bytes >> 20} MiB GPU threshold -- CPU is faster"
    elif gpu_cfg.enabled and not fits_gpu:
        reason = f"exceeds VRAM budget ({pb >> 20} MiB > {gpu_cfg.vram_budget_mib} MiB)"
    else:
        reason = "no usable GPU tier" if cpu == "bitint" else "reference floor"
    return Routing(cpu, reason, pb, fits_gpu)


def _grading_peaks(histogram: Mapping) -> list:
    """Per-grading reduction peaks (bytes) from a histogram keyed by ``(grading, degree) ->
    count``: regroup by grading, then the largest boundary block within each grading."""
    by_grading: dict = defaultdict(dict)
    for (grading, degree), count in histogram.items():
        by_grading[grading][degree] = count
    return [grading_peak_bytes(degrees) for degrees in by_grading.values()]


def dense_reduction_bytes(histogram: Mapping) -> int:
    """Worst-case co-resident reduction memory (bytes) for a batch given as a histogram keyed
    by ``(grading, degree) -> count``: the sum over gradings of each grading's peak block, i.e.
    every grading reduced at once. This is the unbounded-concurrency figure; the compute scheduler
    holds the actual peak below the budget by ordering reductions and spilling held results, so it
    is an upper bound, not the feasibility criterion (see ``max_grading_bytes``).
    """
    return sum(_grading_peaks(histogram))


def max_grading_bytes(histogram: Mapping) -> int:
    """The largest single grading's reduction peak (bytes) over a ``(grading, degree)``
    histogram -- the feasibility floor. Running gradings in deterministic waves, a workload is
    reducible only at budgets at least this large: below it the largest grading cannot fit even
    alone and the scheduler fails loud. At or above it the rest pack into waves under the budget.
    """
    return max(_grading_peaks(histogram), default=0)
