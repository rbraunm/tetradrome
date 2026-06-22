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
    GPU,
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

from tetradrome.scheduler.executor import (                            # noqa: E402
    WarmWorker, _augment_for_admission, _overhead_probe_main, _start_context)
from tetradrome.scheduler.routing import Execution                     # noqa: E402
from tetradrome.scheduler.hostplatform import for_host                    # noqa: E402

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
    worker = WarmWorker(ctx, cores, None, result_queue, setup=warm_setup,
                        between=warm_between, platform=for_host())
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
    worker = WarmWorker(ctx, frozenset(os.sched_getaffinity(0)), None, result_queue,
                        platform=for_host())
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


# -- GPU execution routing wired through the loop (fabricated device, CPU stand-in callables) --

def _gpu_machine():
    cores = frozenset(os.sched_getaffinity(0))
    return Machine(nodes=(NumaNode(0, cores, 8 * _GIB),),
                   gpus=(GPU(index=0, vram_bytes=_GIB, numa_node=0),),
                   mem_cap_bytes=8 * _GIB)


def _gpu(vram):
    return ComputePath(Placement.GPU, cores=1, ram_bytes=_SMALL, vram_bytes=vram)


def test_small_gpu_jobs_route_warm_and_serialize_in_one_worker():
    # three small-vram GPU jobs, uncalibrated, so the gate sends them warm. The hooked warm
    # worker's between-count climbs 0,1,2 across one reused process; if any had gone fresh it
    # would have reset to 0, so distinct climbing counts prove warm + serial + single process.
    jobs = [Job(key=k, run=report_warm_state, inputs={}, paths=(_gpu(_GIB // 10),), cost=1000)
            for k in ("a", "b", "c")]
    report = Scheduler(_gpu_machine(), warm_setup=warm_setup, warm_between=warm_between,
                       context_vram_reserve=0).run(JobGraph(jobs))
    setups = {report.results[k][0] for k in ("a", "b", "c")}
    counts = sorted(report.results[k][1] for k in ("a", "b", "c"))
    assert setups == {True}             # every job ran in the worker that ran setup once
    assert counts == [0, 1, 2]          # one process, serial, freed between each


def test_big_gpu_job_routes_fresh():
    # vram at 60% of the budget trips the firm trigger; a fresh worker never runs warm_setup, so
    # the state it reports is the module default rather than the hooked-worker's True.
    job = Job(key="big", run=report_warm_state, inputs={},
              paths=(_gpu(6 * _GIB // 10),), cost=1000)
    report = Scheduler(_gpu_machine(), warm_setup=warm_setup, warm_between=warm_between,
                       context_vram_reserve=0).run(JobGraph([job]))
    assert report.results["big"] == (False, 0)


def test_calibration_accumulates_a_gpu_rate():
    jobs = [Job(key=k, run=produce, inputs={"value": k}, paths=(_gpu(_GIB // 10),), cost=1000)
            for k in ("a", "b")]
    report = Scheduler(_gpu_machine()).run(JobGraph(jobs))
    assert report.results == {"a": "a", "b": "b"}
    rate = report.calibration.rate(Placement.GPU)
    assert rate is not None and rate > 0.0          # observed seconds/cost folded in


def test_failing_warm_job_does_not_kill_the_shared_worker():
    # two independent components, both small so both route warm; one fails. The shared worker
    # must still complete the other, proving one component's failure stays isolated.
    bad = Job(key="bad", run=boom, inputs={}, paths=(_gpu(_GIB // 10),), cost=1000)
    good = Job(key="good", run=produce, inputs={"value": 42}, paths=(_gpu(_GIB // 10),), cost=1000)
    report = Scheduler(_gpu_machine(), warm_setup=warm_setup, warm_between=warm_between,
                       context_vram_reserve=0).run(JobGraph([bad, good]))
    assert report.results.get("good") == 42
    assert "bad" in report.cancelled
    assert any(failed == "bad" for _component, failed, _err in report.failures)


# -- per-process overhead in the admission view (pure, no processes spawned) --

_CTX = 100 * _SMALL          # 100 MiB: what a CUDA context costs in VRAM
_BASE = 500 * _SMALL         # 500 MiB: what a spawned worker's cold imports cost in RAM


def _aug(paths, *, shared, gpu_budget=_GIB, predicted=None):
    return _augment_for_admission(
        paths, worker_shared=shared, ram_baseline=_BASE, context_vram=_CTX,
        gpu_budget=gpu_budget, predicted_gpu_time=predicted,
        context_overhead=float("inf"), vram_fraction=0.5, time_multiple=10.0)


def test_spawn_charges_baseline_to_a_fresh_cpu_path():
    cpu = ComputePath(Placement.CPU_PINNED, cores=2, ram_bytes=_SMALL)
    (aug,), gpu_exec = _aug((cpu,), shared=False)
    assert aug.ram_bytes == _SMALL + _BASE      # the cold-import baseline a spawned worker pays
    assert aug.vram_bytes == 0
    assert gpu_exec is None                     # no GPU path present


def test_fork_charges_no_cpu_baseline():
    cpu = ComputePath(Placement.CPU_PINNED, cores=2, ram_bytes=_SMALL)
    (aug,), _ = _aug((cpu,), shared=True)
    assert aug.ram_bytes == _SMALL              # forked workers share the imports; nothing added


def test_fresh_gpu_path_charges_context_and_baseline():
    big = _gpu(6 * _GIB // 10)                  # 60% of the 1 GiB budget -> routes fresh
    (aug,), gpu_exec = _aug((big,), shared=False)
    assert gpu_exec is Execution.FRESH
    assert aug.vram_bytes == big.vram_bytes + _CTX     # its own CUDA context
    assert aug.ram_bytes == big.ram_bytes + _BASE      # plus the spawn baseline


def test_warm_gpu_path_adds_no_overhead():
    small = _gpu(_GIB // 10)                     # 10% of budget -> routes warm
    (aug,), gpu_exec = _aug((small,), shared=False)
    assert gpu_exec is Execution.WARM
    assert aug.vram_bytes == small.vram_bytes    # shares the standing warm worker's context
    assert aug.ram_bytes == small.ram_bytes      # and its baseline


def test_fresh_gpu_under_fork_charges_context_only():
    big = _gpu(6 * _GIB // 10)
    (aug,), gpu_exec = _aug((big,), shared=True)
    assert gpu_exec is Execution.FRESH
    assert aug.vram_bytes == big.vram_bytes + _CTX
    assert aug.ram_bytes == big.ram_bytes        # fork shares imports; no RAM baseline


def test_no_device_leaves_gpu_path_untouched():
    small = _gpu(_GIB // 10)
    (aug,), gpu_exec = _aug((small,), shared=False, gpu_budget=None)
    assert gpu_exec is None
    assert aug.vram_bytes == small.vram_bytes and aug.ram_bytes == small.ram_bytes


def test_overhead_probe_reports_real_private_memory():
    # The probe runs a real worker process and reads back a positive private footprint; with no
    # accelerator it reports zero context VRAM. That round-trip is what the per-process charge
    # depends on. (The charge itself is zero under fork, where imports are shared.)
    ctx = _start_context()
    q = ctx.Queue()
    proc = ctx.Process(target=_overhead_probe_main, args=(q, for_host(), None, None))
    proc.start()
    try:
        measured = q.get(timeout=120)
    finally:
        proc.join()
    assert measured is not None
    ram, context_vram = measured
    assert isinstance(ram, int) and ram > 0
    assert context_vram == 0
