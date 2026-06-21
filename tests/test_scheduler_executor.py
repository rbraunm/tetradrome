# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Integration tests for the executor: real spawn workers on a machine built from real cores.

The job callables are module-level so they pickle to spawned workers. DAG ordering is forced by
dependencies, so these are deterministic regardless of how many cores the host actually has.
"""
import os
import time

from tetradrome.scheduler import (
    ComputePath,
    Job,
    JobGraph,
    Machine,
    NumaNode,
    Placement,
    Scheduler,
)

_GIB = 1 << 30
_SMALL = 1 << 20


def _machine():
    # Real cores, so os.sched_setaffinity in the worker always gets a valid mask.
    cores = frozenset(os.sched_getaffinity(0))
    return Machine(nodes=(NumaNode(0, cores, 8 * _GIB),), gpus=(), mem_cap_bytes=8 * _GIB)


def _cpu():
    return ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=_SMALL)


# -- module-level job callables (picklable for spawn) --

def produce(inputs, deps):
    return inputs["value"]


def passthrough(inputs, deps):
    return sum(deps.values()) + inputs["add"]


def boom(inputs, deps):
    raise RuntimeError("kaboom")


def nap(inputs, deps):
    time.sleep(inputs["seconds"])
    return "rested"


def _job(key, run, inputs, deps=()):
    return Job(key=key, run=run, inputs=inputs, paths=(_cpu(),), dependencies=deps)


def test_single_job_runs_and_returns():
    job = _job("a", produce, {"value": 42})
    report = Scheduler(_machine()).run(JobGraph([job]))
    assert report.results == {"a": 42}
    assert report.failures == []


def test_dependency_results_flow_down_a_chain():
    jobs = [
        _job("gen", produce, {"value": 10}),
        _job("mid", passthrough, {"add": 5}, deps=("gen",)),    # 10 + 5
        _job("end", passthrough, {"add": 1}, deps=("mid",)),    # 15 + 1
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.results["end"] == 16


def test_fan_out_then_fan_in():
    jobs = [
        _job("gen", produce, {"value": 3}),
        _job("r1", passthrough, {"add": 1}, deps=("gen",)),     # 4
        _job("r2", passthrough, {"add": 2}, deps=("gen",)),     # 5
        _job("asm", passthrough, {"add": 0}, deps=("r1", "r2")),  # 4 + 5
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.results["asm"] == 9
    assert report.failures == []


def test_failure_abandons_its_component_only():
    jobs = [
        _job("genX", produce, {"value": 1}),
        _job("boom", boom, {}, deps=("genX",)),
        _job("endX", passthrough, {"add": 0}, deps=("boom",)),
        _job("genY", produce, {"value": 7}),
        _job("midY", passthrough, {"add": 2}, deps=("genY",)),   # 9, independent component
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.results.get("midY") == 9                       # other component finishes
    for key in ("genX", "boom", "endX"):
        assert key not in report.results                         # whole X component dropped
    assert {"genX", "boom", "endX"} <= report.cancelled
    assert len(report.failures) == 1
    component, failed_key, _ = report.failures[0]
    assert failed_key == "boom"
    assert component == {"genX", "boom", "endX"}


def test_job_too_big_for_machine_fails_its_component():
    big = Job(key="big", run=produce, inputs={"value": 1},
              paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=1 << 40),))
    report = Scheduler(_machine()).run(JobGraph([big]))
    assert "big" not in report.results
    assert len(report.failures) == 1
    assert report.failures[0][1] == "big"


def test_report_records_run_time():
    # the recorded time covers the run itself: at least the sleep, and not wildly more, since
    # worker startup and affinity setup happen outside the measured window
    seconds = 0.2
    job = Job(key="n", run=nap, inputs={"seconds": seconds}, paths=(_cpu(),), cost=1000.0)
    report = Scheduler(_machine()).run(JobGraph([job]))
    assert report.results["n"] == "rested"
    assert seconds <= report.timings["n"] < seconds + 1.0


def test_failed_job_carries_no_timing():
    report = Scheduler(_machine()).run(JobGraph([_job("boom", boom, {})]))
    assert "boom" not in report.timings        # a job that did not finish has no usable runtime
