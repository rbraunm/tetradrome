# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The placement decision: for one ready job, choose the fastest path the machine can serve.

Capability versus contention is the core distinction. A path is *capable* if the machine could
ever serve it -- the resource type exists and total capacity, less the reserved margin, could
satisfy it -- regardless of what is busy now. The scheduler runs the fastest capable path:

- if its resources are free now, admit it there (carrying a degradation note when a faster path
  was skipped because it is not capable on this box -- never because it was merely busy);
- if its resources exist but are busy, wait for them (contention never degrades a job);
- if no declared path is capable, the job is infeasible and fails loud.

``margin`` is a fraction of each node's RAM and of the cap held back from scheduling, so that
several jobs under-predicting at once still have slack before anything approaches a real limit.
It is folded into both capability and fit so a job that needs more than the schedulable share
is reported infeasible rather than waiting on memory that policy will never hand out. The
function is pure: it reads the ledger and returns a Decision; the loop mutates the ledger on
ADMIT.
"""
from __future__ import annotations

import dataclasses
import enum

from ..errors import TetradromeError
from .inventory import Machine
from .job import ComputePath, Job, Placement
from .ledger import Ledger


class Outcome(enum.Enum):
    ADMIT = "admit"
    WAIT = "wait"


class InfeasibilityAxis(enum.Enum):
    """Why one declared path cannot be served by the bare machine."""
    NO_DEVICE = "no_device"          # the placement's hardware is absent (a GPU path, no GPU)
    EXCEEDS_VRAM = "exceeds_vram"    # the path's VRAM is larger than any device
    EXCEEDS_RAM = "exceeds_ram"      # the path's RAM is larger than the schedulable ceiling
    EXCEEDS_CORES = "exceeds_cores"  # the path needs more cores than exist


@dataclasses.dataclass(frozen=True)
class PathGap:
    """One reason a path is unservable on the bare machine: which path, which axis, and the
    numbers (needed versus available) so the message says exactly what to change. ``available`` is
    0 for NO_DEVICE, since there is no such device at all."""
    placement: Placement
    axis: InfeasibilityAxis
    needed: int
    available: int


def _human_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _fmt_gap(gap: PathGap) -> str:
    if gap.axis is InfeasibilityAxis.NO_DEVICE:
        return f"{gap.placement.value} {gap.axis.value}: no device for this placement"
    if gap.axis is InfeasibilityAxis.EXCEEDS_CORES:
        return (f"{gap.placement.value} {gap.axis.value}: needs {gap.needed} cores, "
                f"has {gap.available}")
    return (f"{gap.placement.value} {gap.axis.value}: needs {_human_bytes(gap.needed)}, "
            f"has {_human_bytes(gap.available)}")


class InfeasibleJobError(Exception):
    """A job no declared path of which the bare machine can serve. Carries one PathGap per path so
    the message names exactly what to change. The scheduler collects these and reports them; it
    does not raise them itself, so one infeasible job does not sink an otherwise feasible batch. A
    caller wanting strictness can raise one from the report."""

    def __init__(self, job_key, gaps):
        self.job_key = job_key
        self.gaps = tuple(gaps)
        detail = "; ".join(_fmt_gap(gap) for gap in self.gaps)
        super().__init__(f"job {job_key!r} infeasible: {detail}")


@dataclasses.dataclass(frozen=True)
class Placed:
    """A concrete admission: the chosen path, the cores granted, and where it runs."""
    path: ComputePath
    cores: frozenset
    node_index: int | None = None
    gpu_index: int | None = None
    note: str | None = None        # capability-degradation note, or None


@dataclasses.dataclass(frozen=True)
class Decision:
    outcome: Outcome
    placed: Placed | None = None
    reason: str | None = None


def _capability_gap(machine: Machine, path: ComputePath, margin: float) -> "PathGap | None":
    """None if the bare machine could ever serve this path; else a typed PathGap saying why. This
    is capability, not contention: it reads only the machine's totals less the reserved margin,
    never the ledger, so the verdict is independent of what is running now."""
    schedulable_cap = machine.mem_cap_bytes - int(margin * machine.mem_cap_bytes)
    if path.placement is Placement.GPU:
        if not machine.gpus:
            return PathGap(path.placement, InfeasibilityAxis.NO_DEVICE, path.vram_bytes, 0)
        best_vram = max(gpu.vram_bytes for gpu in machine.gpus)
        if best_vram < path.vram_bytes:
            return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_VRAM,
                           path.vram_bytes, best_vram)
        if machine.total_cores < path.cores:
            return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_CORES,
                           path.cores, machine.total_cores)
        if schedulable_cap < path.ram_bytes:
            return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_RAM,
                           path.ram_bytes, schedulable_cap)
        return None
    if path.placement is Placement.CPU_PINNED:
        if schedulable_cap < path.ram_bytes:
            return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_RAM,
                           path.ram_bytes, schedulable_cap)
        best_node_cores = max((len(node.cores) for node in machine.nodes), default=0)
        if best_node_cores < path.cores:
            return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_CORES,
                           path.cores, best_node_cores)
        fits = any(len(node.cores) >= path.cores
                   and node.ram_bytes - int(margin * node.ram_bytes) >= path.ram_bytes
                   for node in machine.nodes)
        if not fits:
            best_node_ram = max((node.ram_bytes - int(margin * node.ram_bytes)
                                 for node in machine.nodes), default=0)
            return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_RAM,
                           path.ram_bytes, best_node_ram)
        return None
    # CPU_UNPINNED
    if machine.total_cores < path.cores:
        return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_CORES,
                       path.cores, machine.total_cores)
    if schedulable_cap < path.ram_bytes:
        return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_RAM,
                       path.ram_bytes, schedulable_cap)
    return None


def _take(cores: frozenset, count: int) -> frozenset:
    """A deterministic count-core subset (lowest ids) of a free core set."""
    return frozenset(sorted(cores)[:count])


def _place_now(machine: Machine, ledger: Ledger, path: ComputePath,
               note: str | None, margin: float) -> Placed | None:
    """A concrete Placed if the path's resources are free right now, else None."""
    global_margin = int(margin * machine.mem_cap_bytes)
    if path.placement is Placement.GPU:
        if ledger.global_free_ram() - global_margin < path.ram_bytes:
            return None
        free = ledger.free_cores_all()
        if len(free) < path.cores:
            return None
        for gpu in machine.gpus:
            if ledger.free_vram(gpu.index) >= path.vram_bytes:
                return Placed(path=path, cores=_take(free, path.cores),
                              gpu_index=gpu.index, note=note)
        return None
    if path.placement is Placement.CPU_PINNED:
        if ledger.global_free_ram() - global_margin < path.ram_bytes:
            return None
        for node in machine.nodes:
            free = ledger.free_cores(node.index)
            node_margin = int(margin * node.ram_bytes)
            if (len(free) >= path.cores
                    and ledger.free_ram_node(node.index) - node_margin >= path.ram_bytes):
                return Placed(path=path, cores=_take(free, path.cores),
                              node_index=node.index, note=note)
        return None
    # CPU_UNPINNED
    if ledger.global_free_ram() - global_margin < path.ram_bytes:
        return None
    free = ledger.free_cores_all()
    if len(free) < path.cores:
        return None
    return Placed(path=path, cores=_take(free, path.cores), note=note)


def _gap_summary(skipped: list) -> str:
    return "; ".join(_fmt_gap(gap) for gap in skipped)


def job_feasibility(machine: Machine, job: Job, margin: float = 0.0) -> "InfeasibleJobError | None":
    """The pre-flight verdict for one job: an InfeasibleJobError if the bare machine can serve no
    declared path, else None. The error carries one PathGap per path so the report says exactly
    why. Pass the job with its admission-augmented paths, so the verdict matches what the loop will
    actually try to place: a job that fits bare but not once its own per-process overhead is folded
    in could never be admitted, and must be caught here rather than stalling the loop."""
    gaps = []
    for path in job.paths:
        gap = _capability_gap(machine, path, margin)
        if gap is None:
            return None                  # at least one path is servable -> the job is feasible
        gaps.append(gap)
    return InfeasibleJobError(job.key, tuple(gaps))


def _degradation_note(chosen: ComputePath, skipped: list) -> str | None:
    if not skipped:
        return None
    return (f"ran on {chosen.placement.value}; faster path(s) unavailable on this machine: "
            f"{_gap_summary(skipped)}")


def plan_placement(machine: Machine, ledger: Ledger, job: Job, margin: float = 0.0) -> Decision:
    """Decide where (if anywhere) ``job`` should run, given current ledger state and margin."""
    skipped: list = []
    chosen = None
    for path in job.paths:
        gap = _capability_gap(machine, path, margin)
        if gap is None:
            chosen = path
            break
        skipped.append(gap)
    if chosen is None:
        # Unreachable: the pre-flight feasibility pass removes every job the machine cannot serve
        # before the loop runs, so a job reaching placement always has a capable path. If this
        # fires, feasibility and placement have diverged, which is a scheduler bug, not a user
        # error, so fail loud rather than silently waiting forever.
        raise TetradromeError(
            f"job {job.key!r} reached placement with no capable path ({_gap_summary(skipped)}); "
            f"the feasibility pre-flight should have excluded it")
    note = _degradation_note(chosen, skipped)
    placed = _place_now(machine, ledger, chosen, note, margin)
    if placed is not None:
        return Decision(Outcome.ADMIT, placed=placed)
    return Decision(
        Outcome.WAIT,
        reason=f"job {job.key!r} waiting for {chosen.placement.value} resources",
    )
