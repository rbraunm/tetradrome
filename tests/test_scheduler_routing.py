# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for GPU execution routing: the EWMA calibration and the warm/fresh decision.

Both are pure logic with no device, so the calibration is checked against hand-computed EWMA
values and the decision is exercised at each trigger and on the boundaries.
"""
import pytest

from tetradrome.scheduler import Calibration, Execution, Placement, route_execution


# -- Calibration --

def test_rate_unknown_until_observed():
    cal = Calibration()
    assert cal.rate(Placement.GPU) is None
    assert cal.predicted_time(1000, Placement.GPU) is None


def test_first_observation_seeds_the_rate():
    cal = Calibration()
    cal.observe(cost=1000, placement=Placement.GPU, seconds=2.0)
    assert cal.rate(Placement.GPU) == pytest.approx(2.0 / 1000)
    # predicted time recovers exactly for a job of the same cost
    assert cal.predicted_time(1000, Placement.GPU) == pytest.approx(2.0)


def test_ewma_blends_subsequent_observations():
    cal = Calibration(alpha=0.3)
    cal.observe(cost=100, placement=Placement.GPU, seconds=1.0)    # rate 0.01
    cal.observe(cost=100, placement=Placement.GPU, seconds=3.0)    # observed 0.03
    expected = 0.3 * 0.03 + 0.7 * 0.01
    assert cal.rate(Placement.GPU) == pytest.approx(expected)


def test_zero_cost_observation_is_ignored():
    cal = Calibration()
    cal.observe(cost=0, placement=Placement.GPU, seconds=5.0)
    assert cal.rate(Placement.GPU) is None


def test_rates_are_per_placement():
    cal = Calibration()
    cal.observe(cost=100, placement=Placement.GPU, seconds=1.0)
    assert cal.rate(Placement.GPU) is not None
    assert cal.rate(Placement.CPU_PINNED) is None


def test_alpha_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        Calibration(alpha=0.0)
    with pytest.raises(ValueError):
        Calibration(alpha=1.5)


# -- route_execution --

_BUDGET = 1 << 30          # 1 GiB device budget
_OVERHEAD = 0.2            # 200 ms to stand up a context


def test_big_vram_goes_fresh_regardless_of_time():
    # 60% of budget, and even with no predicted time the vram trigger fires
    decision = route_execution(predicted_vram=int(0.6 * _BUDGET), vram_budget=_BUDGET,
                               predicted_time=None, context_overhead=_OVERHEAD)
    assert decision is Execution.FRESH


def test_small_vram_uncalibrated_goes_warm():
    # small footprint, no calibration yet -> time trigger held off -> warm
    decision = route_execution(predicted_vram=int(0.1 * _BUDGET), vram_budget=_BUDGET,
                               predicted_time=None, context_overhead=_OVERHEAD)
    assert decision is Execution.WARM


def test_small_vram_long_job_goes_fresh():
    # small footprint but the run is long enough to amortize a fresh context
    decision = route_execution(predicted_vram=int(0.1 * _BUDGET), vram_budget=_BUDGET,
                               predicted_time=5.0, context_overhead=_OVERHEAD)  # 5s >> 10*0.2
    assert decision is Execution.FRESH


def test_small_vram_quick_job_goes_warm():
    decision = route_execution(predicted_vram=int(0.1 * _BUDGET), vram_budget=_BUDGET,
                               predicted_time=0.5, context_overhead=_OVERHEAD)  # 0.5s < 2.0s
    assert decision is Execution.WARM


def test_vram_trigger_boundary_is_inclusive():
    # exactly the fraction counts as big
    decision = route_execution(predicted_vram=_BUDGET // 2, vram_budget=_BUDGET,
                               predicted_time=None, context_overhead=_OVERHEAD,
                               vram_fraction=0.5)
    assert decision is Execution.FRESH


def test_time_trigger_boundary_is_inclusive():
    # predicted_time exactly time_multiple * overhead counts as long
    decision = route_execution(predicted_vram=0, vram_budget=_BUDGET,
                               predicted_time=10.0 * _OVERHEAD, context_overhead=_OVERHEAD,
                               time_multiple=10.0)
    assert decision is Execution.FRESH


def test_zero_budget_fails_loud():
    with pytest.raises(ValueError):
        route_execution(predicted_vram=0, vram_budget=0,
                        predicted_time=None, context_overhead=_OVERHEAD)
