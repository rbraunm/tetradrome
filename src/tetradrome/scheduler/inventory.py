# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Machine inventory: per-NUMA-node cores and RAM, GPUs, and the real memory ceiling.

The scheduler places work against this, so it must reflect the box as the OS actually presents
it: cores and RAM per NUMA node (so a job can be pinned where both its cores and its memory are
local), the true ceiling on total memory, and any CUDA devices. Everything above this layer is
platform-agnostic and consumes the resulting Machine.

Topology discovery is host-specific and lives behind the NumaTopology class: for_host() returns
the implementation for the OS we are on. Linux reads /sys and the cgroup. Windows is a skeleton
that fails loud until it is implemented against a real Windows NUMA host. Parsing is split into
pure functions so the fiddly /sys formats are tested directly.
"""
from __future__ import annotations

import abc
import dataclasses
import os
import sys

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
    """The cores this process may actually run on (honours cpuset / taskset). Linux-internal:
    only the Linux topology calls it, and Linux always provides sched_getaffinity."""
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


# ---- host-specific topology --------------------------------------------

class NumaTopology(abc.ABC):
    """Host-specific discovery of the schedulable topology: the NUMA nodes (the cores this
    process may use on each, and that node's RAM) and the ceiling on total resident memory.

    Instantiated per host by ``for_host``; everything above this layer consumes the resulting
    Machine and never branches on platform. A new OS is supported by adding a subclass and a
    branch in ``for_host`` -- the contract is just the two methods here.
    """

    @abc.abstractmethod
    def nodes(self) -> tuple[NumaNode, ...]:
        """The NUMA nodes. A non-NUMA box is one node spanning all allowed cores and all RAM."""

    @abc.abstractmethod
    def mem_cap_bytes(self, physical_total_bytes: int) -> int:
        """The ceiling on total resident memory: a container limit capped by physical RAM, or
        physical RAM when nothing tighter applies."""


class LinuxNumaTopology(NumaTopology):
    """Topology from /sys and the cgroup, as the Linux kernel and container present it."""

    def nodes(self) -> tuple[NumaNode, ...]:
        return detect_nodes()

    def mem_cap_bytes(self, physical_total_bytes: int) -> int:
        return detect_mem_ceiling(physical_total_bytes)


class WindowsNumaTopology(NumaTopology):
    """Topology on Windows. Not implemented yet: it needs a real Windows NUMA host to build and
    test against, and modelling a Windows box as a single node without one would be a guess
    dressed up as fact. The Win32 each method needs is noted; until it is in and tested on a
    Windows platform, both fail loud rather than return a number nobody verified.
    """

    def nodes(self) -> tuple[NumaNode, ...]:
        # TODO(windows-numa): GetLogicalProcessorInformationEx(RelationNumaNode) for the nodes
        # and each node's processor mask, intersected with the process affinity mask from
        # GetProcessAffinityMask; per-node RAM via GetNumaAvailableMemoryNodeEx. ctypes against
        # kernel32, no third-party dependency.
        raise NotImplementedError(
            "Windows NUMA topology detection is not implemented yet; it needs a Windows NUMA "
            "test platform to build against.")

    def mem_cap_bytes(self, physical_total_bytes: int) -> int:
        # TODO(windows-numa): GlobalMemoryStatusEx.ullTotalPhys for physical RAM. Windows has no
        # cgroup; a job-object memory limit (QueryInformationJobObject) is the analogue when one
        # applies, otherwise physical RAM is the ceiling.
        raise NotImplementedError(
            "Windows memory ceiling detection is not implemented yet; it needs a Windows test "
            "platform to build against.")


def for_host() -> NumaTopology:
    """The topology discovery for the OS we are running on."""
    if sys.platform == "win32":
        return WindowsNumaTopology()
    return LinuxNumaTopology()


def detect_machine() -> Machine:
    """Probe the box into a schedulable inventory via the host's topology discovery."""
    topology = for_host()
    nodes = topology.nodes()
    physical_total = sum(node.ram_bytes for node in nodes)
    return Machine(nodes=nodes, gpus=detect_gpus(),
                   mem_cap_bytes=topology.mem_cap_bytes(physical_total))
