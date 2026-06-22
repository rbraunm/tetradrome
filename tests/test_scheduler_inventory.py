# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for machine inventory: the /sys parsers (exact) and live discovery (structural)."""
import pytest

from tetradrome.scheduler import Machine, NumaNode, detect_machine
from tetradrome.scheduler.inventory import (
    DebianHostPlatform,
    UbuntuHostPlatform,
    WindowsHostPlatform,
    for_host,
    parse_cgroup_limit,
    parse_cpulist,
    parse_meminfo_total,
    parse_node_meminfo_total,
    tightest_ceiling,
)


def test_parse_cpulist_ranges_and_singletons():
    assert parse_cpulist("0-3,8,12-13") == frozenset({0, 1, 2, 3, 8, 12, 13})


def test_parse_cpulist_single():
    assert parse_cpulist("5") == frozenset({5})


def test_parse_cpulist_empty():
    assert parse_cpulist("") == frozenset()
    assert parse_cpulist("  \n") == frozenset()


def test_parse_cpulist_dedupes_overlap():
    assert parse_cpulist("0-2,1-3") == frozenset({0, 1, 2, 3})


def test_parse_node_meminfo_total():
    text = "Node 0 MemTotal:       16384 kB\nNode 0 MemFree:        4096 kB\n"
    assert parse_node_meminfo_total(text) == 16384 * 1024


def test_parse_node_meminfo_total_missing_raises():
    with pytest.raises(ValueError):
        parse_node_meminfo_total("Node 0 MemFree: 4096 kB\n")


def test_parse_meminfo_total():
    assert parse_meminfo_total("MemTotal:  32768 kB\nMemFree: 1024 kB\n") == 32768 * 1024


def test_parse_cgroup_limit_real_value():
    assert parse_cgroup_limit("200\n") == 200


def test_parse_cgroup_limit_max_is_none():
    assert parse_cgroup_limit("max\n") is None


def test_tightest_ceiling_container_cap_binds():
    # In a container, physical is the host node-sum (300), but /proc/meminfo is virtualized to
    # the 200 cap and the cgroup agrees. The cap, not host RAM, is the real ceiling.
    assert tightest_ceiling(physical_total_bytes=300, meminfo_total_bytes=200,
                            cgroup_limit_bytes=200) == 200


def test_tightest_ceiling_meminfo_binds_without_cgroup():
    # cgroup limit absent ('max' -> None), but the virtualized /proc/meminfo still carries the
    # container cap below host RAM.
    assert tightest_ceiling(physical_total_bytes=300, meminfo_total_bytes=200,
                            cgroup_limit_bytes=None) == 200


def test_tightest_ceiling_bare_metal_is_physical():
    # No container: the three sources agree, cgroup unlimited. Physical RAM is the ceiling.
    assert tightest_ceiling(physical_total_bytes=300, meminfo_total_bytes=300,
                            cgroup_limit_bytes=None) == 300


def test_tightest_ceiling_cgroup_sentinel_dropped_by_min():
    # A cgroup-v1 'unlimited' sentinel sits far above physical; min() ignores it without a
    # special case, so physical (matching meminfo on bare metal) remains the ceiling.
    assert tightest_ceiling(physical_total_bytes=300, meminfo_total_bytes=300,
                            cgroup_limit_bytes=9223372036854771712) == 300


def test_machine_aggregates():
    m = Machine(
        nodes=(NumaNode(0, frozenset({0, 1}), 100), NumaNode(1, frozenset({2, 3}), 200)),
        gpus=(),
        mem_cap_bytes=250,
    )
    assert m.total_cores == 4
    assert m.total_node_ram_bytes == 300


def test_detect_machine_smoke():
    # Environment-tolerant: structure and basic sanity, not exact values.
    m = detect_machine()
    assert len(m.nodes) >= 1
    assert m.total_cores >= 1
    assert all(node.cores for node in m.nodes)
    assert m.mem_cap_bytes > 0
    assert m.total_node_ram_bytes > 0

# -- host-dispatched topology --

# -- host platform --

def test_for_host_picks_the_right_platform():
    import sys
    platform = for_host()
    if sys.platform == "win32":
        assert isinstance(platform, WindowsHostPlatform)
    else:
        # our container is Ubuntu; debian-family would be the shim, anything else would raise
        assert isinstance(platform, UbuntuHostPlatform)
        assert platform.name in ("ubuntu", "debian")


def test_windows_platform_constructs_and_names_itself_anywhere():
    # constructible on any OS since the ctypes calls are lazy; only its primitives need Windows
    assert WindowsHostPlatform().name == "windows"


def test_windows_platform_primitives_on_windows():
    import os
    import sys
    if sys.platform != "win32":
        pytest.skip("windows primitives run on windows")
    platform = WindowsHostPlatform()
    nodes = platform.nodes()
    assert len(nodes) >= 1
    assert all(node.cores for node in nodes)
    assert platform.mem_cap_bytes(sum(n.ram_bytes for n in nodes)) > 0
    assert platform.private_bytes(os.getpid()) > 0
    assert platform.private_bytes(2 ** 31) is None
    platform.pin(nodes[0].cores)


def test_debian_is_the_ubuntu_shim_but_names_itself_debian():
    # reuses Ubuntu's implementation (it is a subclass) but reports 'debian' so logs show the shim
    platform = DebianHostPlatform()
    assert isinstance(platform, UbuntuHostPlatform)
    assert platform.name == "debian"
    assert UbuntuHostPlatform().name == "ubuntu"


def test_ubuntu_platform_topology_and_primitives():
    import os
    import sys
    if sys.platform == "win32":
        pytest.skip("ubuntu platform test runs on linux")
    platform = UbuntuHostPlatform()
    nodes = platform.nodes()
    assert len(nodes) >= 1
    assert all(node.cores for node in nodes)
    assert platform.mem_cap_bytes(sum(n.ram_bytes for n in nodes)) > 0
    # sampling our own process reports some private memory; a dead pid reports None
    assert platform.private_bytes(os.getpid()) > 0
    assert platform.private_bytes(2 ** 31) is None
    platform.pin(os.sched_getaffinity(0))      # pinning to our own allowed set is a no-op-safe call
