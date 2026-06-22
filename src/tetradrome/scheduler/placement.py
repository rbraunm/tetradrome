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

from .inventory import Machine
from .job import ComputePath, Job, Placement
from .ledger import Ledger


class Outcome(enum.Enum):
    ADMIT = "admit"
    WAIT = "wait"
    INFEASIBLE = "infeasible"


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


def _capability_gap(machine: Machine, path: ComputePath, margin: float) -> str | None:
    """None if the machine could ever serve this path; else why it cannot (a capability gap)."""
    schedulable_cap = machine.mem_cap_bytes - int(margin * machine.mem_cap_bytes)
    if path.placement is Placement.GPU:
        if not machine.gpus:
            return "no GPU on this machine"
        if not any(gpu.vram_bytes >= path.vram_bytes for gpu in machine.gpus):
            return "no GPU with enough VRAM"
        if machine.total_cores < path.cores:
            return "not enough host cores"
        if schedulable_cap < path.ram_bytes:
            return "exceeds the memory ceiling"
        return None
    if path.placement is Placement.CPU_PINNED:
        if not any(len(node.cores) >= path.cores
                   and node.ram_bytes - int(margin * node.ram_bytes) >= path.ram_bytes
                   for node in machine.nodes):
            return "no single node has enough cores and RAM"
        if schedulable_cap < path.ram_bytes:
            return "exceeds the memory ceiling"
        return None
    # CPU_UNPINNED
    if machine.total_cores < path.cores:
        return "not enough cores across all nodes"
    if schedulable_cap < path.ram_bytes:
        return "exceeds the memory ceiling"
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
    return "; ".join(f"{path.placement.value} ({why})" for path, why in skipped)


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
        skipped.append((path, gap))
    if chosen is None:
        return Decision(
            Outcome.INFEASIBLE,
            reason=f"job {job.key!r} has no compute path this machine can run: "
                   f"{_gap_summary(skipped)}",
        )
    note = _degradation_note(chosen, skipped)
    placed = _place_now(machine, ledger, chosen, note, margin)
    if placed is not None:
        return Decision(Outcome.ADMIT, placed=placed)
    return Decision(
        Outcome.WAIT,
        reason=f"job {job.key!r} waiting for {chosen.placement.value} resources",
    )
