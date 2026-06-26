# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the machine inventory: the Machine data model and live composition via
detect_machine (structural, environment-tolerant)."""
from tetradrome.scheduler import Machine, NumaNode, detect_machine


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
