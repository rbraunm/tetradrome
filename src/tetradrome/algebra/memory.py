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


def predict_size(cx) -> ComplexSize:
    """Exact size facts for a GradedComplex, including the packed reduction peak."""
    degrees = tuple(cx.degrees())
    dims = {n: cx.dim(n) for n in degrees}
    total = sum(dims.values())
    peak = 0
    largest = (0, 0, 0)
    for n in degrees:
        cols = dims[n]
        rows = cx.dim(n + 1)
        if cols == 0 or rows == 0:
            continue
        nwords = (rows + 63) // 64
        # during reduction the packed reducer holds the whole matrix plus the pivots
        # (at most min(cols, rows) columns), each column nwords uint64 words.
        b = (cols + min(cols, rows)) * nwords * 8
        if b > peak:
            peak, largest = b, (n, cols, rows)
    return ComplexSize(degrees, dims, total, largest, peak)


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
