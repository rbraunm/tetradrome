# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The resource ledger: what is committed right now, and what is free.

Free capacity is derived from the running allocations, never tracked as a separate mutable
total that could drift. Each allocation is charged the worse of its declared peak and its
measured actual usage (``max(peak, actual)``): a job still ramping toward its peak is charged
the full peak, so its low current usage cannot trick us into over-admitting, and a job that
has exceeded its prediction is charged its real usage, so a low estimate cannot either.

Memory binds on two axes at once: per NUMA node (a node-pinned job's RAM must fit that node's
physical RAM) and globally (the sum across nodes must fit the cgroup ceiling). Cores are
tracked as concrete ids so a placement can pin to them.
"""
from __future__ import annotations

import dataclasses

from .inventory import Machine
from .job import Placement


@dataclasses.dataclass
class Allocation:
    """A running job's hold on resources. ``actual_*`` are updated by the live sampler; the
    ledger charges ``max(declared, actual)``. ``node_index`` is the node a pinned job's cores
    and RAM belong to (None when unpinned); ``gpu_index`` is set for a GPU placement."""
    job_key: object
    placement: Placement
    cores: frozenset
    declared_ram: int
    node_index: int | None = None
    gpu_index: int | None = None
    declared_vram: int = 0
    actual_ram: int = 0
    actual_vram: int = 0

    @property
    def charged_ram(self) -> int:
        return max(self.declared_ram, self.actual_ram)

    @property
    def charged_vram(self) -> int:
        return max(self.declared_vram, self.actual_vram)


class Ledger:
    """Live accounting against a fixed machine. Pure reads; the scheduler loop mutates it via
    add/remove/set_actual as jobs start, finish, and are sampled."""

    def __init__(self, machine: Machine):
        self.machine = machine
        self._nodes = {node.index: node for node in machine.nodes}
        self._gpus = {gpu.index: gpu for gpu in machine.gpus}
        self._allocs: dict = {}

    # ---- mutation ----

    def add(self, alloc: Allocation) -> None:
        if alloc.job_key in self._allocs:
            raise ValueError(f"job {alloc.job_key!r} is already allocated")
        self._allocs[alloc.job_key] = alloc

    def remove(self, job_key) -> Allocation:
        return self._allocs.pop(job_key)

    def set_actual(self, job_key, ram_bytes: int, vram_bytes: int = 0) -> None:
        alloc = self._allocs[job_key]
        alloc.actual_ram = ram_bytes
        alloc.actual_vram = vram_bytes

    def allocations(self) -> tuple:
        return tuple(self._allocs.values())

    # ---- cores ----

    def _allocated_cores(self) -> set:
        used: set = set()
        for alloc in self._allocs.values():
            used |= alloc.cores
        return used

    def free_cores(self, node_index: int) -> frozenset:
        return self._nodes[node_index].cores - self._allocated_cores()

    def free_cores_all(self) -> frozenset:
        allocated = self._allocated_cores()
        free: set = set()
        for node in self._nodes.values():
            free |= (node.cores - allocated)
        return frozenset(free)

    # ---- memory (per node and global) ----

    def committed_ram_node(self, node_index: int) -> int:
        return sum(a.charged_ram for a in self._allocs.values() if a.node_index == node_index)

    def free_ram_node(self, node_index: int) -> int:
        return self._nodes[node_index].ram_bytes - self.committed_ram_node(node_index)

    def global_committed_ram(self) -> int:
        return sum(a.charged_ram for a in self._allocs.values())

    def global_free_ram(self) -> int:
        return self.machine.mem_cap_bytes - self.global_committed_ram()

    # ---- vram (per device) ----

    def committed_vram(self, gpu_index: int) -> int:
        return sum(a.charged_vram for a in self._allocs.values() if a.gpu_index == gpu_index)

    def free_vram(self, gpu_index: int) -> int:
        return self._gpus[gpu_index].vram_bytes - self.committed_vram(gpu_index)
