# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the placement decision: capability vs contention, degradation notes, gating."""
import pytest

from tetradrome.scheduler import (
    GPU,
    Allocation,
    ComputePath,
    InfeasibilityAxis,
    Job,
    Ledger,
    Machine,
    NumaNode,
    Outcome,
    Placement,
    job_feasibility,
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
    error = job_feasibility(m, _job("j", (_gpu(1, 10, 8),)))
    assert error is not None
    assert error.job_key == "j"
    assert len(error.gaps) == 1
    assert error.gaps[0].axis is InfeasibilityAxis.NO_DEVICE


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


def test_margin_makes_over_schedulable_job_infeasible():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    # job needs 98; margin 3% of 100 = 3 -> schedulable 97 -> no path the bare machine can serve
    error = job_feasibility(m, _job("j", (_pinned(1, 98),)), margin=0.03)
    assert error is not None
    assert error.gaps[0].axis is InfeasibilityAxis.EXCEEDS_RAM
    assert error.gaps[0].available == 97


def test_margin_zero_admits_to_full_capacity():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    d = plan_placement(m, Ledger(m), _job("j", (_pinned(1, 98),)), margin=0.0)
    assert d.outcome is Outcome.ADMIT


def test_margin_leaves_headroom_when_it_fits():
    m = _machine(nodes=((0, {0, 1}, 100),), cap=100)
    d = plan_placement(m, Ledger(m), _job("j", (_pinned(1, 90),)), margin=0.03)
    assert d.outcome is Outcome.ADMIT          # 90 <= 97 schedulable


def _inline(ram):
    return ComputePath(Placement.INLINE, 1, ram)


def test_inline_admits_on_ram_alone_and_takes_no_core():
    m = _machine()
    ledger = Ledger(m)
    for node in m.nodes:                            # every core is busy
        ledger.add(Allocation(("busy", node.index), Placement.CPU_PINNED, node.cores, 0, node.index))
    d = plan_placement(m, ledger, _job("j", (_inline(50),)))
    assert d.outcome is Outcome.ADMIT
    assert d.placed.cores == frozenset()
    assert d.placed.node_index is None


def test_inline_waits_while_ram_is_committed():
    m = _machine()                                  # cap 250
    ledger = Ledger(m)
    ledger.add(Allocation("held", Placement.CPU_UNPINNED, frozenset(), 220))
    assert plan_placement(m, ledger, _job("j", (_inline(50),))).outcome is Outcome.WAIT


def test_inline_beyond_the_cap_is_infeasible():
    error = job_feasibility(_machine(), _job("j", (_inline(251),)))
    assert [gap.axis for gap in error.gaps] == [InfeasibilityAxis.EXCEEDS_RAM]


def test_inline_path_runs_on_one_thread():
    with pytest.raises(ValueError, match="cores must be 1"):
        ComputePath(Placement.INLINE, 2, 10)


# -- which capable path: measurement where there is one, declared order otherwise --

def _figures(table):
    """An estimate function from {placement: (run_seconds, dispatch_seconds)}."""
    return lambda path: table[path.placement]


def _both(estimate=None, order=(Placement.INLINE, Placement.CPU_PINNED)):
    m = _machine()
    paths = tuple(_inline(10) if placement is Placement.INLINE else _pinned(1, 10)
                  for placement in order)
    return plan_placement(m, Ledger(m), _job("j", paths), estimate=estimate).placed.path.placement


def test_unmeasured_job_takes_its_first_declared_path():
    assert _both() is Placement.INLINE
    assert _both(order=(Placement.CPU_PINNED, Placement.INLINE)) is Placement.CPU_PINNED
    unknown_run = _figures({Placement.INLINE: (None, 0.0), Placement.CPU_PINNED: (None, 0.05)})
    assert _both(unknown_run, order=(Placement.CPU_PINNED, Placement.INLINE)) is Placement.CPU_PINNED
    unknown_dispatch = _figures({Placement.INLINE: (0.001, 0.0), Placement.CPU_PINNED: (0.001, None)})
    assert _both(unknown_dispatch, order=(Placement.CPU_PINNED, Placement.INLINE)) is Placement.CPU_PINNED


def test_job_shorter_than_a_dispatch_runs_inline_even_if_declared_second():
    quick = _figures({Placement.INLINE: (0.01, 0.0), Placement.CPU_PINNED: (0.01, 0.05)})
    assert _both(quick, order=(Placement.CPU_PINNED, Placement.INLINE)) is Placement.INLINE


def test_job_longer_than_a_dispatch_goes_to_a_worker_even_if_inline_is_declared_first():
    long = _figures({Placement.INLINE: (2.0, 0.0), Placement.CPU_PINNED: (2.0, 0.05)})
    assert _both(long) is Placement.CPU_PINNED


def test_run_time_equal_to_the_dispatch_time_stays_inline():
    even = _figures({Placement.INLINE: (0.05, 0.0), Placement.CPU_PINNED: (0.05, 0.05)})
    assert _both(even) is Placement.INLINE


def test_among_worker_paths_the_smallest_run_plus_dispatch_wins():
    m = _machine(gpus=((0, 100),))
    job = _job("j", (_gpu(1, 10, 10), _pinned(1, 10)))
    slow_gpu = _figures({Placement.GPU: (1.0, 3.0), Placement.CPU_PINNED: (2.0, 0.05)})
    fast_gpu = _figures({Placement.GPU: (0.1, 0.5), Placement.CPU_PINNED: (2.0, 0.05)})
    half_known = _figures({Placement.GPU: (None, 3.0), Placement.CPU_PINNED: (2.0, 0.05)})
    assert plan_placement(m, Ledger(m), job, estimate=slow_gpu).placed.path.placement is Placement.CPU_PINNED
    assert plan_placement(m, Ledger(m), job, estimate=fast_gpu).placed.path.placement is Placement.GPU
    assert plan_placement(m, Ledger(m), job, estimate=half_known).placed.path.placement is Placement.GPU


def test_chosen_worker_path_still_waits_rather_than_falling_back_inline():
    m = _machine()
    ledger = Ledger(m)
    for node in m.nodes:                            # every core is busy
        ledger.add(Allocation(("busy", node.index), Placement.CPU_PINNED, node.cores, 0, node.index))
    long = _figures({Placement.INLINE: (2.0, 0.0), Placement.CPU_PINNED: (2.0, 0.05)})
    job = _job("j", (_inline(10), _pinned(1, 10)))
    assert plan_placement(m, ledger, job, estimate=long).outcome is Outcome.WAIT


# -- hyperthreads: a physical core each first, second threads only where measured to help --

def _smt_machine():
    """Two nodes of two physical cores, each core two hyperthreads: (0, 4), (1, 5) | (2, 6), (3, 7)."""
    return Machine(
        nodes=(NumaNode(0, frozenset({0, 1, 4, 5}), 100, (frozenset({0, 4}), frozenset({1, 5}))),
               NumaNode(1, frozenset({2, 3, 6, 7}), 100, (frozenset({2, 6}), frozenset({3, 7})))),
        gpus=(), mem_cap_bytes=200,
    )


def _kind_job(key, kind, cores=1):
    return Job(key=key, run=lambda i, d: None, inputs=None, paths=(_pinned(cores, 1),), kind=kind)


def _admit(machine, ledger, job, shares=None):
    decision = plan_placement(machine, ledger, job, shares=shares)
    if decision.outcome is Outcome.ADMIT:
        ledger.add(Allocation(job_key=job.key, placement=Placement.CPU_PINNED,
                              cores=decision.placed.cores, declared_ram=1,
                              node_index=decision.placed.node_index, kind=job.kind))
        return sorted(decision.placed.cores)
    return None


def test_jobs_take_a_physical_core_each_across_nodes_before_any_second_thread():
    machine = _smt_machine()
    ledger = Ledger(machine)
    taken = [_admit(machine, ledger, _kind_job(i, "shares"), shares=lambda kind: True)
             for i in range(8)]
    assert taken == [[0], [1], [2], [3], [4], [5], [6], [7]]


def test_without_a_sharing_figure_a_job_waits_for_a_core_of_its_own():
    machine = _smt_machine()
    ledger = Ledger(machine)
    taken = [_admit(machine, ledger, _kind_job(i, "unmeasured")) for i in range(5)]
    assert taken == [[0], [1], [2], [3], None]


def test_a_kind_that_does_not_share_never_takes_a_second_thread():
    machine = _smt_machine()
    ledger = Ledger(machine)
    shares = {"shares": True, "alone": False}.get
    for i in range(4):
        _admit(machine, ledger, _kind_job(i, "shares"), shares)
    assert _admit(machine, ledger, _kind_job("late", "alone"), shares) is None


def test_a_second_thread_is_not_taken_beside_a_kind_that_does_not_share():
    machine = _smt_machine()
    ledger = Ledger(machine)
    shares = {"shares": True, "alone": False}.get
    kinds = ["alone", "shares", "alone", "alone"]              # on cores (0,4) (1,5) (2,6) (3,7)
    for i, kind in enumerate(kinds):
        _admit(machine, ledger, _kind_job(i, kind), shares)
    assert _admit(machine, ledger, _kind_job("a", "shares"), shares) == [5]
    assert _admit(machine, ledger, _kind_job("b", "shares"), shares) is None


def test_a_job_wider_than_the_physical_cores_takes_its_own_second_threads():
    machine = _smt_machine()
    assert _admit(machine, Ledger(machine), _kind_job("wide", "alone", cores=3),
                  shares=lambda kind: False) == [0, 1, 4]


def test_a_node_without_listed_physical_cores_treats_each_cpu_as_one():
    machine = _machine()                                        # NumaNode(...) with no physical
    ledger = Ledger(machine)
    taken = [_admit(machine, ledger, _kind_job(i, "unmeasured")) for i in range(5)]
    assert taken == [[0], [1], [2], [3], None]
