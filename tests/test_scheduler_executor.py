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


# -- warm worker: persistent, serial, frees between jobs --

from tetradrome.scheduler.executor import WarmWorker, _start_context   # noqa: E402

# State lives in the worker process. setup flips it once; between increments after each job. A
# fresh process per job would reset it, so an accumulating count is proof of process reuse.
_warm_state = {"setup": False, "betweens": 0}


def warm_setup():
    _warm_state["setup"] = True


def warm_between():
    _warm_state["betweens"] += 1


def report_warm_state(inputs, deps):
    return (_warm_state["setup"], _warm_state["betweens"])


def test_warm_worker_runs_jobs_serially_in_one_process():
    ctx = _start_context()
    result_queue = ctx.Queue()
    cores = frozenset(os.sched_getaffinity(0))
    worker = WarmWorker(ctx, cores, None, result_queue, setup=warm_setup, between=warm_between)
    try:
        for key in ("a", "b", "c"):
            worker.dispatch(key, report_warm_state, {}, {})
        results = {}
        for _ in range(3):
            key, status, payload, _seconds = result_queue.get(timeout=30)
            assert status == "ok"
            results[key] = payload
    finally:
        worker.shutdown()
    # setup ran once before any job; between ran once per finished job, so the count climbs 0,1,2.
    # A new process each time would have shown (True, 0) all three times.
    assert results["a"] == (True, 0)
    assert results["b"] == (True, 1)
    assert results["c"] == (True, 2)


def test_warm_worker_survives_a_failing_job():
    ctx = _start_context()
    result_queue = ctx.Queue()
    worker = WarmWorker(ctx, frozenset(os.sched_getaffinity(0)), None, result_queue)
    try:
        worker.dispatch("bad", boom, {}, {})
        worker.dispatch("good", produce, {"value": 7}, {})
        outcomes = {}
        for _ in range(2):
            key, status, payload, _seconds = result_queue.get(timeout=30)
            outcomes[key] = (status, payload)
    finally:
        worker.shutdown()
    assert outcomes["bad"][0] == "error"
    assert outcomes["good"] == ("ok", 7)        # the worker kept serving after the failure
