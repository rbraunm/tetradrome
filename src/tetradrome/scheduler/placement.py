# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The placement decision: for one ready job, choose the path it runs on, then admit or wait.

Capability versus contention is the core distinction. A path is *capable* if the machine could
ever serve it -- the resource type exists and total capacity, less the reserved margin, could
satisfy it -- regardless of what is busy now. The scheduler picks one capable path:

- if its resources are free now, admit it there (carrying a degradation note when a path
  declared faster was skipped because it is not capable on this box -- never because it was
  merely busy);
- if its resources exist but are busy, wait for them (contention never degrades a job);
- if no declared path is capable, the job is infeasible and fails loud.

Which capable path is picked follows measurement where there is one. A job runs inline when its
predicted run time is no more than what dispatching a worker costs, because the inline call is
then done before the worker would have started; past that an inline call only holds up the loop
and every other job, so a worker path wins. Among worker paths the smallest predicted
run-plus-dispatch time wins. Wherever a figure is still unmeasured the declared order decides:
the builder lists a job's paths fastest first.

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
    # For a WAIT: True when the chosen path found no CPUs for it on any node, whatever the memory.
    # CPUs only get scarcer while a pass admits jobs, so a like job need not be tried again in it.
    short_of_cpus: bool = False


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
    if path.placement is Placement.INLINE:
        if schedulable_cap < path.ram_bytes:
            return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_RAM,
                           path.ram_bytes, schedulable_cap)
        return None
    # CPU_UNPINNED
    if machine.total_cores < path.cores:
        return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_CORES,
                       path.cores, machine.total_cores)
    if schedulable_cap < path.ram_bytes:
        return PathGap(path.placement, InfeasibilityAxis.EXCEEDS_RAM,
                       path.ram_bytes, schedulable_cap)
    return None


def _pick_cpus(nodes, ledger: Ledger, count: int, kind: str, shares,
               whole_only: bool = False) -> "frozenset | None":
    """``count`` CPUs on ``nodes`` for a job of ``kind``, or None if they are not free now.

    One CPU of each wholly free physical core comes first, lowest first. Unless ``whole_only``,
    the rest are second threads: those of the cores just taken, which slows no other job, and
    those of busy cores when ``shares`` says both this kind and every kind already on the core do
    more sharing a core than one of them does alone."""
    free = frozenset().union(*(ledger.free_cores(node.index) for node in nodes))
    physical = [core for node in nodes for core in node.physical_cores]
    whole = [core for core in physical if core <= free][:count]
    chosen = [min(core) for core in whole]
    if len(chosen) == count:
        return frozenset(chosen)
    if whole_only:
        return None
    second = sorted(cpu for core in whole for cpu in core if cpu != min(core))
    if shares is not None and shares(kind):
        for core in physical:
            idle = core & free
            if idle and idle != core and all(shares(other) for other in ledger.kinds_on(core)):
                second += sorted(idle)
    chosen += second[:count - len(chosen)]
    return frozenset(chosen) if len(chosen) == count else None


def _place_now(machine: Machine, ledger: Ledger, path: ComputePath, note: str | None,
               margin: float, kind: str = "", shares=None) -> tuple:
    """``(placed, short_of_cpus)``: a concrete Placed if the path's resources are free right now,
    else None with whether CPUs were what it lacked, on every node and whatever the memory."""
    enough_ram = ledger.global_free_ram() - int(margin * machine.mem_cap_bytes) >= path.ram_bytes
    if path.placement is Placement.INLINE:
        # The scheduler's own thread: no worker core to grant, only its RAM to fit.
        return (Placed(path=path, cores=frozenset(), note=note) if enough_ram else None), False
    if path.placement is Placement.CPU_PINNED:
        found_cpus = False
        # A wholly free core on any node is better than a second thread on the first node.
        for whole_only in (True, False):
            for node in machine.nodes:
                cpus = _pick_cpus((node,), ledger, path.cores, kind, shares, whole_only)
                if cpus is None:
                    continue
                found_cpus = True
                node_free = ledger.free_ram_node(node.index) - int(margin * node.ram_bytes)
                if enough_ram and node_free >= path.ram_bytes:
                    return Placed(path=path, cores=cpus, node_index=node.index, note=note), False
        return None, not found_cpus
    cpus = _pick_cpus(machine.nodes, ledger, path.cores, kind, shares)
    if cpus is None:
        return None, True
    if not enough_ram:
        return None, False
    if path.placement is Placement.CPU_UNPINNED:
        return Placed(path=path, cores=cpus, note=note), False
    for gpu in machine.gpus:
        if ledger.free_vram(gpu.index) >= path.vram_bytes:
            return Placed(path=path, cores=cpus, gpu_index=gpu.index, note=note), False
    return None, False


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


def _choose(capable: list, estimate) -> ComputePath:
    """The path to run on, from the capable ones in declared order. ``estimate(path)`` returns
    ``(run_seconds, dispatch_seconds)``, either None while unmeasured."""
    if estimate is None or len(capable) == 1:
        return capable[0]
    figures = [estimate(path) for path in capable]
    workers = [(path, run, dispatch) for path, (run, dispatch) in zip(capable, figures)
               if path.placement is not Placement.INLINE]
    inline = [(path, run) for path, (run, _) in zip(capable, figures)
              if path.placement is Placement.INLINE]
    worker = None
    worker_dispatch = None
    if workers:
        worker, _, worker_dispatch = workers[0]
        if all(run is not None and dispatch is not None for _, run, dispatch in workers):
            worker, _, worker_dispatch = min(workers, key=lambda item: item[1] + item[2])
    if not inline:
        return worker
    inline_path, inline_run = inline[0]
    if worker is None:
        return inline_path
    if inline_run is None or worker_dispatch is None:
        return capable[0]
    return inline_path if inline_run <= worker_dispatch else worker


def plan_placement(machine: Machine, ledger: Ledger, job: Job, margin: float = 0.0,
                   estimate=None, shares=None) -> Decision:
    """Decide where (if anywhere) ``job`` should run, given current ledger state and margin.
    ``estimate``, when given, maps a path to its measured ``(run_seconds, dispatch_seconds)``.
    ``shares``, when given, says whether a kind of job may take the second thread of a busy
    physical core; without it a job waits for a core of its own."""
    skipped: list = []
    capable: list = []
    for path in job.paths:
        gap = _capability_gap(machine, path, margin)
        if gap is None:
            capable.append(path)
        elif not capable:
            skipped.append(gap)         # a declared-faster path this machine cannot serve
    chosen = _choose(capable, estimate) if capable else None
    if chosen is None:
        # Unreachable: the pre-flight feasibility pass removes every job the machine cannot serve
        # before the loop runs, so a job reaching placement always has a capable path. If this
        # fires, feasibility and placement have diverged, which is a scheduler bug, not a user
        # error, so fail loud rather than silently waiting forever.
        raise TetradromeError(
            f"job {job.key!r} reached placement with no capable path ({_gap_summary(skipped)}); "
            f"the feasibility pre-flight should have excluded it")
    note = _degradation_note(chosen, skipped)
    placed, short_of_cpus = _place_now(machine, ledger, chosen, note, margin, job.kind, shares)
    if placed is not None:
        return Decision(Outcome.ADMIT, placed=placed)
    return Decision(
        Outcome.WAIT,
        reason=f"job {job.key!r} waiting for {chosen.placement.value} resources",
        short_of_cpus=short_of_cpus,
    )
