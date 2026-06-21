# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the placement decision: capability vs contention, degradation notes, gating."""
from tetradrome.scheduler import (
    GPU,
    Allocation,
    ComputePath,
    Job,
    Ledger,
    Machine,
    NumaNode,
    Outcome,
    Placement,
    plan_placement,
)


def _machine(nodes=((0, {0, 1}, 100), (1, {2, 3}, 200)), gpus=(), cap=250):
    return Machine(
        nodes=tuple(NumaNode(i, frozenset(c), r) for i, c, r in nodes),
        gpus=tuple(GPU(i, v, None) for i, v in gpus),
        mem_cap_bytes=cap,
    )


def _job(key, paths):
    return Job(key=key, run=lambda i, d: None, inputs=None, paths=paths)


def _pinned(cores, ram):
    return ComputePath(Placement.CPU_PINNED, cores, ram)


def _unpinned(cores, ram):
    return ComputePath(Placement.CPU_UNPINNED, cores, ram)


def _gpu(cores, ram, vram):
    return ComputePath(Placement.GPU, cores, ram, vram)


def test_admit_pinned_fits():
    m = _machine()
    d = plan_placement(m, Ledger(m), _job("j", (_pinned(1, 50),)))
    assert d.outcome is Outcome.ADMIT
    assert d.placed.node_index in (0, 1)
    assert len(d.placed.cores) == 1
    assert d.placed.note is None


def test_too_big_for_any_node_degrades_to_unpinned_with_note():
    m = _machine()                                  # nodes 100/200; cap 250
    j = _job("j", (_pinned(1, 250), _unpinned(1, 250)))
    d = plan_placement(m, Ledger(m), j)
    assert d.outcome is Outcome.ADMIT
    assert d.placed.path.placement is Placement.CPU_UNPINNED
    assert d.placed.node_index is None
    assert "cpu_pinned" in d.placed.note            # recorded the capability gap


def test_gpu_only_on_gpuless_machine_is_infeasible():
    m = _machine(gpus=())
    d = plan_placement(m, Ledger(m), _job("j", (_gpu(1, 10, 8),)))
    assert d.outcome is Outcome.INFEASIBLE
    assert "no GPU" in d.reason


def test_gpu_unavailable_degrades_to_cpu_with_note():
    m = _machine(gpus=())
    j = _job("j", (_gpu(1, 10, 8), _pinned(1, 10)))
    d = plan_placement(m, Ledger(m), j)
    assert d.outcome is Outcome.ADMIT
    assert d.placed.path.placement is Placement.CPU_PINNED
    assert "gpu" in d.placed.note


def test_capable_but_busy_waits():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    led = Ledger(m)
    led.add(Allocation("a", Placement.CPU_PINNED, frozenset({0, 1}), 10, node_index=0))
    d = plan_placement(m, led, _job("j", (_pinned(1, 10),)))
    assert d.outcome is Outcome.WAIT                 # capable (2 cores total) but 0 free now


def test_contention_never_degrades_to_slower_path():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    led = Ledger(m)
    led.add(Allocation("a", Placement.CPU_PINNED, frozenset({0, 1}), 10, node_index=0))
    # fastest pinned is capable-but-busy; a slower unpinned exists, but contention waits.
    d = plan_placement(m, led, _job("j", (_pinned(1, 10), _unpinned(1, 10))))
    assert d.outcome is Outcome.WAIT


def test_max_peak_actual_gates_admission_then_frees():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    led = Ledger(m)
    led.add(Allocation("a", Placement.CPU_PINNED, frozenset({0}), declared_ram=80, node_index=0))
    j = _job("j", (_pinned(1, 50),))
    # charged max(80, 0) = 80 -> only 20 free -> WAIT (low actual must not over-admit)
    assert plan_placement(m, led, j).outcome is Outcome.WAIT
    led.set_actual("a", ram_bytes=10)               # still below peak -> still charged 80
    assert plan_placement(m, led, j).outcome is Outcome.WAIT
    led.remove("a")
    assert plan_placement(m, led, j).outcome is Outcome.ADMIT


def test_underprediction_charged_at_actual():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    led = Ledger(m)
    led.add(Allocation("a", Placement.CPU_PINNED, frozenset({0}), declared_ram=10, node_index=0))
    led.set_actual("a", ram_bytes=90)               # reality exceeds the low estimate
    d = plan_placement(m, led, _job("j", (_pinned(1, 50),)))
    assert d.outcome is Outcome.WAIT                 # charged 90, not 10


def test_gpu_admits_on_capable_machine():
    m = _machine(gpus=((0, 16),))
    d = plan_placement(m, Ledger(m), _job("j", (_gpu(1, 10, 8),)))
    assert d.outcome is Outcome.ADMIT
    assert d.placed.gpu_index == 0
    assert d.placed.note is None


def test_margin_blocks_admission_near_capacity():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    # job needs 98; margin 3% of 100 = 3 -> schedulable 97 -> infeasible
    d = plan_placement(m, Ledger(m), _job("j", (_pinned(1, 98),)), margin=0.03)
    assert d.outcome is Outcome.INFEASIBLE


def test_margin_zero_admits_to_full_capacity():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    d = plan_placement(m, Ledger(m), _job("j", (_pinned(1, 98),)), margin=0.0)
    assert d.outcome is Outcome.ADMIT


def test_margin_leaves_headroom_when_it_fits():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    d = plan_placement(m, Ledger(m), _job("j", (_pinned(1, 90),)), margin=0.03)
    assert d.outcome is Outcome.ADMIT          # 90 <= 97 schedulable
