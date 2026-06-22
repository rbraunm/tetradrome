# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Machine inventory: the schedulable description of one box -- per-NUMA-node cores and RAM,
GPUs, and the true ceiling on total memory.

This is the data model the scheduler places work against, plus ``detect_machine``, which
assembles it by composing the two probe axes: the host platform (cores, RAM, ceiling) and the
accelerator (GPU devices). Both axes live in their own modules; this one stays free of
OS-specific and vendor-specific code so it imports cleanly anywhere.
"""
from __future__ import annotations

import dataclasses

@dataclasses.dataclass(frozen=True)
class NumaNode:
    """One NUMA node: the cores we are allowed to run on there, and its RAM."""
    index: int
    cores: frozenset[int]
    ram_bytes: int


@dataclasses.dataclass(frozen=True)
class GPU:
    """One CUDA device. ``numa_node`` is the node it is attached to, or None if not known."""
    index: int
    vram_bytes: int
    numa_node: int | None = None


@dataclasses.dataclass(frozen=True)
class Machine:
    """The schedulable inventory of one box.

    ``mem_cap_bytes`` is the ceiling on TOTAL resident memory (the cgroup limit, or physical
    RAM when there is no tighter limit). Per-node ``ram_bytes`` is the physical RAM of each
    node. The scheduler must respect both: per-node usage <= that node's RAM, and the sum over
    nodes <= mem_cap_bytes.
    """
    nodes: tuple[NumaNode, ...]
    gpus: tuple[GPU, ...]
    mem_cap_bytes: int

    @property
    def total_cores(self) -> int:
        return sum(len(node.cores) for node in self.nodes)

    @property
    def total_node_ram_bytes(self) -> int:
        return sum(node.ram_bytes for node in self.nodes)

    def describe(self) -> str:
        lines = [f"machine: {len(self.nodes)} NUMA node(s), {self.total_cores} cores, "
                 f"memory ceiling {self.mem_cap_bytes >> 30} GiB"]
        for node in self.nodes:
            lines.append(f"  node {node.index}: {len(node.cores)} cores, "
                         f"{node.ram_bytes >> 30} GiB")
        for g in self.gpus:
            where = f"node {g.numa_node}" if g.numa_node is not None else "node unknown"
            lines.append(f"  gpu {g.index}: {g.vram_bytes >> 30} GiB VRAM, {where}")
        return "\n".join(lines)


def detect_machine() -> "Machine":
    """Probe the box into a schedulable inventory by composing the host-platform and accelerator
    axes. Imported locally so the data model above stays free of axis dependencies."""
    from .hostplatform import for_host
    from .accelerator import detect_accelerator
    platform = for_host()
    nodes = platform.nodes()
    physical_total = sum(node.ram_bytes for node in nodes)
    accelerator = detect_accelerator()
    gpus = accelerator.detect_devices() if accelerator is not None else ()
    return Machine(nodes=nodes, gpus=gpus,
                   mem_cap_bytes=platform.mem_cap_bytes(physical_total))
