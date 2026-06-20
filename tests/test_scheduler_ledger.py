# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the resource ledger: free derivation, per-node vs global RAM, max(peak,actual)."""
import pytest

from tetradrome.scheduler import GPU, Allocation, Ledger, Machine, NumaNode, Placement


def _machine(nodes=((0, {0, 1}, 100), (1, {2, 3}, 200)), gpus=(), cap=250):
    return Machine(
        nodes=tuple(NumaNode(i, frozenset(c), r) for i, c, r in nodes),
        gpus=tuple(GPU(i, v, None) for i, v in gpus),
        mem_cap_bytes=cap,
    )


def _pinned(job_key, node_index, cores, ram):
    return Allocation(job_key=job_key, placement=Placement.CPU_PINNED,
                      cores=frozenset(cores), declared_ram=ram, node_index=node_index)


def test_empty_ledger_all_free():
    led = Ledger(_machine())
    assert led.free_cores(0) == frozenset({0, 1})
    assert led.free_ram_node(0) == 100
    assert led.free_ram_node(1) == 200
    assert led.global_free_ram() == 250
    assert led.free_cores_all() == frozenset({0, 1, 2, 3})


def test_pinned_alloc_consumes_node_and_global():
    led = Ledger(_machine())
    led.add(_pinned("a", node_index=0, cores={0}, ram=30))
    assert led.free_cores(0) == frozenset({1})
    assert led.committed_ram_node(0) == 30
    assert led.free_ram_node(0) == 70
    assert led.committed_ram_node(1) == 0          # other node untouched
    assert led.global_committed_ram() == 30
    assert led.global_free_ram() == 220


def test_charge_uses_max_peak_when_actual_lower():
    led = Ledger(_machine())
    led.add(_pinned("a", node_index=0, cores={0}, ram=30))
    led.set_actual("a", ram_bytes=10)              # ramping below the declared peak
    assert led.free_ram_node(0) == 70              # still charged the 30 peak
    assert led.global_free_ram() == 220


def test_charge_uses_actual_when_it_exceeds_peak():
    led = Ledger(_machine())
    led.add(_pinned("a", node_index=0, cores={0}, ram=30))
    led.set_actual("a", ram_bytes=80)              # under-predicted: reality is higher
    assert led.free_ram_node(0) == 20
    assert led.global_free_ram() == 170


def test_unpinned_alloc_hits_global_not_node():
    led = Ledger(_machine())
    led.add(Allocation("u", Placement.CPU_UNPINNED, frozenset({0, 2}), declared_ram=40))
    assert led.committed_ram_node(0) == 0          # not charged to any node pool
    assert led.committed_ram_node(1) == 0
    assert led.global_free_ram() == 210            # but it does hit the global ceiling
    assert led.free_cores(0) == frozenset({1})     # its cores still leave their nodes
    assert led.free_cores(1) == frozenset({3})


def test_vram_per_device():
    led = Ledger(_machine(gpus=((0, 100), (1, 50))))
    led.add(Allocation("g", Placement.GPU, frozenset({0}), declared_ram=10,
                       gpu_index=0, declared_vram=40))
    assert led.free_vram(0) == 60
    assert led.free_vram(1) == 50                  # other device untouched
    led.set_actual("g", ram_bytes=10, vram_bytes=70)
    assert led.free_vram(0) == 30                  # max(40, 70)


def test_remove_frees():
    led = Ledger(_machine())
    led.add(_pinned("a", node_index=0, cores={0}, ram=30))
    led.remove("a")
    assert led.free_ram_node(0) == 100
    assert led.free_cores(0) == frozenset({0, 1})


def test_duplicate_add_raises():
    led = Ledger(_machine())
    led.add(_pinned("a", node_index=0, cores={0}, ram=30))
    with pytest.raises(ValueError):
        led.add(_pinned("a", node_index=0, cores={1}, ram=30))
