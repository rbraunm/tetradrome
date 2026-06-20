# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Machine inventory: per-NUMA-node cores and RAM, GPUs, and the real memory ceiling.

The scheduler places work against this, so it must reflect the box as the kernel and the
container actually present it: cores and RAM per NUMA node (so a job can be pinned where both
its cores and its memory are local), the cgroup memory cap (the true ceiling on total use,
which in a container is often below the sum of the nodes' RAM), and any CUDA devices.

Discovery reads /sys and the cgroup. A machine that exposes no NUMA topology is modelled as a
single node holding all allowed cores and all RAM -- the correct description of a non-NUMA
box, not a fallback that hides an error. Parsing is split into pure functions so the fiddly
/sys formats are tested directly.
"""
from __future__ import annotations

import dataclasses
import os

_NODE_ROOT = "/sys/devices/system/node"
_CGROUP_MAX = (
    "/sys/fs/cgroup/memory.max",                       # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",     # cgroup v1
)
_MEMINFO = "/proc/meminfo"


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


# ---- pure parsers (tested directly against sample /sys content) ----------

def parse_cpulist(text: str) -> frozenset[int]:
    """A Linux cpulist like '0-3,8,12-13' -> the set of CPU ids. Empty text -> empty set."""
    text = text.strip()
    if not text:
        return frozenset()
    cores: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cores.update(range(int(lo), int(hi) + 1))
        else:
            cores.add(int(part))
    return frozenset(cores)


def parse_node_meminfo_total(text: str) -> int:
    """Bytes of MemTotal from a node's meminfo ('Node N MemTotal:  X kB')."""
    for line in text.splitlines():
        if "MemTotal:" in line:
            return int(line.split("MemTotal:")[1].split()[0]) * 1024
    raise ValueError("no MemTotal line in node meminfo")


def parse_meminfo_total(text: str) -> int:
    """Bytes of MemTotal from /proc/meminfo."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split(":")[1].split()[0]) * 1024
    raise ValueError("no MemTotal line in /proc/meminfo")


def parse_mem_ceiling(cgroup_text: str, physical_total_bytes: int) -> int:
    """The true memory ceiling: the cgroup limit capped by physical RAM. 'max' (cgroup v2) or
    any sentinel above physical means physical RAM is the real ceiling."""
    cgroup_text = cgroup_text.strip()
    if cgroup_text == "max":
        return physical_total_bytes
    return min(int(cgroup_text), physical_total_bytes)


# ---- discovery ----------------------------------------------------------

def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _allowed_cpus() -> frozenset[int]:
    """The cores this process may actually run on (honours cpuset / taskset)."""
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("the scheduler needs Linux CPU affinity (os.sched_getaffinity); "
                           "this OS does not provide it.")
    return frozenset(os.sched_getaffinity(0))


def detect_nodes() -> tuple[NumaNode, ...]:
    """Per-node cores (intersected with our CPU affinity) and RAM, from /sys. Models a single
    node spanning all allowed cores and all RAM when no NUMA topology is exposed."""
    allowed = _allowed_cpus()
    nodes: list[NumaNode] = []
    if os.path.isdir(_NODE_ROOT):
        for name in sorted(os.listdir(_NODE_ROOT)):
            if not name.startswith("node") or not name[4:].isdigit():
                continue
            node_dir = os.path.join(_NODE_ROOT, name)
            cores = parse_cpulist(_read(os.path.join(node_dir, "cpulist"))) & allowed
            if not cores:
                continue                                  # no core here we may use
            ram = parse_node_meminfo_total(_read(os.path.join(node_dir, "meminfo")))
            nodes.append(NumaNode(index=int(name[4:]), cores=cores, ram_bytes=ram))
    if nodes:
        return tuple(nodes)
    # No usable NUMA topology: one node = all allowed cores + all RAM.
    return (NumaNode(index=0, cores=allowed, ram_bytes=parse_meminfo_total(_read(_MEMINFO))),)


def detect_gpus() -> tuple[GPU, ...]:
    """Usable CUDA devices with their VRAM. NUMA affinity is left unknown until GPU placement
    needs it. The hardware detector currently lives under algebra and is imported lazily here,
    so the scheduler stays import-independent of the math layers (the detector ultimately
    belongs in this inventory layer, with algebra consuming it)."""
    from ..algebra import gpu

    info = gpu.detect_gpu()
    if not info.device_usable:
        return ()
    return tuple(
        GPU(index=i, vram_bytes=(dev.memory_total_mib or 0) * 1024 * 1024, numa_node=None)
        for i, dev in enumerate(info.devices)
    )


def detect_mem_ceiling(physical_total_bytes: int) -> int:
    """The total-memory ceiling from the cgroup, capped by physical RAM; physical when there
    is no tighter cgroup limit."""
    for path in _CGROUP_MAX:
        if os.path.exists(path):
            return parse_mem_ceiling(_read(path), physical_total_bytes)
    return physical_total_bytes


def detect_machine() -> Machine:
    """Probe the box into a schedulable inventory."""
    nodes = detect_nodes()
    physical_total = sum(node.ram_bytes for node in nodes)
    return Machine(nodes=nodes, gpus=detect_gpus(),
                   mem_cap_bytes=detect_mem_ceiling(physical_total))
