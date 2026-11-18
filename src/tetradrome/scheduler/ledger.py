# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The resource ledger: what is committed right now, and what is free.

Every allocation goes in and out through ``add``, ``remove``, ``discard`` and ``set_actual``,
which keep running totals beside the allocations, so a query costs the same whether the ledger
holds ten allocations or a hundred thousand. Each allocation is charged the worse of its declared peak and its
measured actual usage (``max(peak, actual)``): a job still ramping toward its peak is charged
the full peak, so its low current usage cannot trick us into over-admitting, and a job that
has exceeded its prediction is charged its real usage, so a low estimate cannot either.

Memory binds on two axes at once: per NUMA node (a node-pinned job's RAM must fit that node's
physical RAM) and globally (the sum across nodes must fit the cgroup ceiling). Cores are
tracked as concrete ids so a placement can pin to them.
"""
from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict

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
    kind: str = ""

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
        self._ram = 0
        self._ram_by_node: dict = defaultdict(int)
        self._vram_by_gpu: dict = defaultdict(int)
        self._core_holders: Counter = Counter()
        self._kind_on: dict = {}        # CPU -> the kind of the job holding it

    # ---- mutation ----

    def _charge(self, alloc: Allocation, sign: int) -> None:
        self._ram += sign * alloc.charged_ram
        self._ram_by_node[alloc.node_index] += sign * alloc.charged_ram
        self._vram_by_gpu[alloc.gpu_index] += sign * alloc.charged_vram
        for core in alloc.cores:
            self._core_holders[core] += sign
            if sign > 0:
                self._kind_on[core] = alloc.kind
            if not self._core_holders[core]:
                del self._core_holders[core]
                del self._kind_on[core]

    def add(self, alloc: Allocation) -> None:
        if alloc.job_key in self._allocs:
            raise ValueError(f"job {alloc.job_key!r} is already allocated")
        self._allocs[alloc.job_key] = alloc
        self._charge(alloc, +1)

    def remove(self, job_key) -> Allocation:
        alloc = self._allocs.pop(job_key)
        self._charge(alloc, -1)
        return alloc

    def discard(self, job_key) -> None:
        """Remove an allocation if present, no error if it is not. For held outputs, whose charge
        may or may not exist when a component is abandoned."""
        if job_key in self._allocs:
            self.remove(job_key)

    def set_actual(self, job_key, ram_bytes: int, vram_bytes: int = 0) -> None:
        alloc = self._allocs[job_key]
        self._charge(alloc, -1)
        alloc.actual_ram = ram_bytes
        alloc.actual_vram = vram_bytes
        self._charge(alloc, +1)

    def allocations(self) -> tuple:
        return tuple(self._allocs.values())

    # ---- cores ----

    def free_cores(self, node_index: int) -> frozenset:
        return self._nodes[node_index].cores - self._core_holders.keys()

    def kinds_on(self, cpus) -> frozenset:
        """The kinds of the jobs holding any of ``cpus``."""
        return frozenset(self._kind_on[cpu] for cpu in cpus if cpu in self._kind_on)

    def free_cores_all(self) -> frozenset:
        free: set = set()
        for node in self._nodes.values():
            free |= (node.cores - self._core_holders.keys())
        return frozenset(free)

    # ---- memory (per node and global) ----

    def committed_ram_node(self, node_index: int) -> int:
        return self._ram_by_node[node_index]

    def free_ram_node(self, node_index: int) -> int:
        return self._nodes[node_index].ram_bytes - self.committed_ram_node(node_index)

    def global_committed_ram(self) -> int:
        return self._ram

    def global_free_ram(self) -> int:
        return self.machine.mem_cap_bytes - self.global_committed_ram()

    # ---- vram (per device) ----

    def committed_vram(self, gpu_index: int) -> int:
        return self._vram_by_gpu[gpu_index]

    def free_vram(self, gpu_index: int) -> int:
        return self._gpus[gpu_index].vram_bytes - self.committed_vram(gpu_index)
