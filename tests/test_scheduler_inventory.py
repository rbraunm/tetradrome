# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for machine inventory: the /sys parsers (exact) and live discovery (structural)."""
import pytest

from tetradrome.scheduler import Machine, NumaNode, detect_machine
from tetradrome.scheduler.inventory import (
    parse_cpulist,
    parse_mem_ceiling,
    parse_meminfo_total,
    parse_node_meminfo_total,
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


def test_parse_mem_ceiling_limit_under_physical_binds():
    assert parse_mem_ceiling("200\n", physical_total_bytes=300) == 200


def test_parse_mem_ceiling_max_uses_physical():
    assert parse_mem_ceiling("max\n", physical_total_bytes=300) == 300


def test_parse_mem_ceiling_sentinel_above_physical_uses_physical():
    # a cgroup-v1 'unlimited' sentinel sits far above physical, so physical is the real ceiling
    assert parse_mem_ceiling("9223372036854771712", physical_total_bytes=300) == 300


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
