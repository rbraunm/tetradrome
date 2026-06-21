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
import logging
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


# ---- host platform: everything OS-specific, one class per host ---------

class HostPlatform(abc.ABC):
    """All OS-specific behavior the scheduler needs, in one place per host: discovering the
    topology, pinning a worker to cores, and reading a process's private memory. ``for_host``
    returns the implementation for the OS we are on, and everything above this layer is
    platform-agnostic. Adding an OS is one new subclass plus a branch in ``for_host``.

    The methods are called from different places -- topology from the main process, ``pin`` from
    inside each worker, ``private_bytes`` from the sampler -- but they are all host-specific, so
    they live together rather than scattered as platform branches across the executor.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identity for logs, so a shimmed or unusual host is visible (e.g. 'ubuntu',
        'debian', 'windows')."""

    @abc.abstractmethod
    def nodes(self) -> tuple[NumaNode, ...]:
        """The NUMA nodes. A non-NUMA box is one node spanning all allowed cores and all RAM."""

    @abc.abstractmethod
    def mem_cap_bytes(self, physical_total_bytes: int) -> int:
        """The ceiling on total resident memory: a container limit capped by physical RAM, or
        physical RAM when nothing tighter applies."""

    @abc.abstractmethod
    def pin(self, cores) -> None:
        """Bind the calling process to ``cores``."""

    @abc.abstractmethod
    def private_bytes(self, pid: int) -> int | None:
        """The private resident bytes of a process, or None if it is gone or unreadable."""


class UbuntuHostPlatform(HostPlatform):
    """The validated Linux platform, exercised on Ubuntu 24.04. Topology from /sys and the
    cgroup, pinning via sched_setaffinity, and private memory from /proc smaps_rollup.
    """

    @property
    def name(self) -> str:
        return "ubuntu"

    def nodes(self) -> tuple[NumaNode, ...]:
        return detect_nodes()

    def mem_cap_bytes(self, physical_total_bytes: int) -> int:
        return detect_mem_ceiling(physical_total_bytes)

    def pin(self, cores) -> None:
        os.sched_setaffinity(0, set(cores))

    def private_bytes(self, pid: int) -> int | None:
        # USS = Private_Clean + Private_Dirty from smaps_rollup. Under forkserver every worker
        # shares the parent's warm pages copy-on-write, so RSS would count that inherited
        # footprint in full in every worker and summing it would phantom-charge memory that
        # physically exists once. Private bytes are what the job actually added on top of the
        # shared base, which is the marginal footprint the ledger should charge.
        private = 0
        try:
            with open(f"/proc/{pid}/smaps_rollup") as handle:
                for line in handle:
                    if line.startswith("Private_Clean:") or line.startswith("Private_Dirty:"):
                        private += int(line.split()[1]) * 1024     # the field is in kB
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            return None
        return private


class DebianHostPlatform(UbuntuHostPlatform):
    """Debian on the Ubuntu shim. Debian is debian-family like Ubuntu and should behave the same,
    so it reuses the Ubuntu implementation wholesale and only reports its name as 'debian', so the
    shimming is visible in logs without anyone having to say the host is Debian. If Debian ever
    needs its own behavior it starts as small overrides here and graduates to a full platform if
    it grows enough to warrant one.
    """

    _warned = False

    def __init__(self) -> None:
        if not DebianHostPlatform._warned:
            DebianHostPlatform._warned = True
            logging.getLogger(__name__).warning(
                "running Debian on the Ubuntu host-platform shim: behavior should match but is "
                "unverified on Debian. Please open an issue for anything that misbehaves so a "
                "real Debian platform can be built out.")

    @property
    def name(self) -> str:
        return "debian"


class WindowsHostPlatform(HostPlatform):
    """Windows. Built next: the Win32 each method needs is noted, implemented with ctypes against
    kernel32 and psapi and validated on a Windows 11 host. Until that lands every method fails
    loud rather than return a number nobody verified.
    """

    @property
    def name(self) -> str:
        return "windows"

    def nodes(self) -> tuple[NumaNode, ...]:
        # TODO(windows): GetLogicalProcessorInformationEx(RelationNumaNode) for the nodes and each
        # node's processor mask, intersected with GetProcessAffinityMask; per-node RAM via
        # GetNumaAvailableMemoryNodeEx.
        raise NotImplementedError("Windows topology is not implemented yet (next commit).")

    def mem_cap_bytes(self, physical_total_bytes: int) -> int:
        # TODO(windows): GlobalMemoryStatusEx.ullTotalPhys; no cgroup, so a job-object limit
        # (QueryInformationJobObject) is the analogue when one applies, else physical RAM.
        raise NotImplementedError("Windows memory ceiling is not implemented yet (next commit).")

    def pin(self, cores) -> None:
        # TODO(windows): SetProcessAffinityMask(GetCurrentProcess(), mask) for up to 64 logical
        # processors; processor groups above that.
        raise NotImplementedError("Windows pinning is not implemented yet (next commit).")

    def private_bytes(self, pid: int) -> int | None:
        # TODO(windows): OpenProcess + GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX.PrivateUsage.
        # Spawn has no copy-on-write sharing, so private usage is already clean per process.
        raise NotImplementedError("Windows memory sampling is not implemented yet (next commit).")


def _os_release_ids() -> tuple[str, str]:
    """(ID, ID_LIKE) from /etc/os-release, or empty strings if it is absent."""
    ids: dict[str, str] = {}
    try:
        with open("/etc/os-release") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    ids[key] = value.strip().strip('"')
    except FileNotFoundError:
        return ("", "")
    return (ids.get("ID", ""), ids.get("ID_LIKE", ""))


def for_host() -> HostPlatform:
    """The host platform for the OS we are running on. Raises loudly on an unrecognized host so
    that supporting it is a deliberate addition, not a silent guess."""
    if sys.platform == "win32":
        return WindowsHostPlatform()
    distro_id, id_like = _os_release_ids()
    if distro_id == "ubuntu":
        return UbuntuHostPlatform()
    if distro_id == "debian" or "debian" in id_like.split():
        return DebianHostPlatform()
    raise RuntimeError(
        f"unsupported host platform (os-release ID={distro_id!r}, ID_LIKE={id_like!r}); add a "
        f"HostPlatform subclass and a branch in for_host().")


def detect_machine() -> Machine:
    """Probe the box into a schedulable inventory via the host platform."""
    platform = for_host()
    nodes = platform.nodes()
    physical_total = sum(node.ram_bytes for node in nodes)
    return Machine(nodes=nodes, gpus=detect_gpus(),
                   mem_cap_bytes=platform.mem_cap_bytes(physical_total))
