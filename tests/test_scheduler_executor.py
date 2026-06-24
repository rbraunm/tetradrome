# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Integration tests for the executor: real spawn workers on a machine built from real cores.

The job callables are module-level so they pickle to spawned workers. DAG ordering is forced by
dependencies, so these are deterministic regardless of how many cores the host actually has.
"""
import os
import time

import pytest

from tetradrome.errors import TetradromeError
from tetradrome.scheduler import (
    ComputePath,
    GPU,
    InfeasibilityAxis,
    Job,
    JobGraph,
    Machine,
    NumaNode,
    Placement,
    Scheduler,
    Shard,
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


def test_job_too_big_for_machine_is_infeasible():
    big = Job(key="big", run=produce, inputs={"value": 1},
              paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=1 << 40),))
    report = Scheduler(_machine()).run(JobGraph([big]))
    assert "big" not in report.results
    assert not report.failures                       # not a runtime failure; caught up front
    assert [e.job_key for e in report.infeasible] == ["big"]
    assert report.infeasible[0].gaps[0].axis is InfeasibilityAxis.EXCEEDS_RAM
    assert "big" in report.cancelled


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


def test_infeasible_job_does_not_sink_the_batch():
    # One feasible component (a -> b) and one independent job whose only path needs more RAM than
    # the whole machine. Pre-flight abandons only the infeasible job's lineage; the feasible
    # component still completes, and the infeasible job is reported, not raised.
    m = _machine()                                       # 8 GiB ceiling
    feasible_a = _job("a", produce, {"value": 7})
    feasible_b = _job("b", passthrough, {"add": 3}, deps=("a",))
    huge = ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=64 * _GIB)
    bad = Job(key="bad", run=produce, inputs={"value": 1}, paths=(huge,))
    report = Scheduler(m).run(JobGraph([feasible_a, feasible_b, bad]))

    assert report.results["b"] == 10                     # feasible component ran to its terminal
    assert "bad" in report.cancelled                     # the infeasible job's lineage abandoned
    assert [e.job_key for e in report.infeasible] == ["bad"]
    assert report.infeasible[0].gaps[0].axis is InfeasibilityAxis.EXCEEDS_RAM


def test_intermediate_results_freed_terminals_kept():
    # A producer consumed by one terminal. After the run, the terminal's result remains but the
    # intermediate's is dropped, freed when its last consumer was dispatched.
    a = _job("a", produce, {"value": 5})
    b = _job("b", passthrough, {"add": 1}, deps=("a",))
    report = Scheduler(_machine()).run(JobGraph([a, b]))
    assert report.results["b"] == 6                  # terminal kept
    assert "a" not in report.results                 # intermediate freed on b's dispatch


def test_output_over_budget_warns(caplog):
    import logging
    # declares a 1-byte output but returns a list that pickles far larger: the measured result
    # exceeds the declaration, so the same kind of warning the working set throws fires for tuning.
    big_output = Job(key="x", run=produce, inputs={"value": list(range(1000))},
                     paths=(_cpu(),), output_bytes=1)
    with caplog.at_level(logging.WARNING):
        Scheduler(_machine()).run(JobGraph([big_output]))
    assert any("output exceeded declared budget" in r.message for r in caplog.records)


def _small_machine(cap):
    cores = frozenset(os.sched_getaffinity(0))
    return Machine(nodes=(NumaNode(0, cores, cap),), gpus=(), mem_cap_bytes=cap)


def _producer(key, value):
    # 1 MiB working set, but a declared 2 MiB held result -- the result is what builds up.
    return Job(key=key, run=produce, inputs={"value": value},
               paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=_SMALL),),
               output_bytes=2 * _SMALL)


def test_spill_relieves_output_pressure_and_restores():
    # Three producers whose held 2 MiB results (6 MiB) overflow a 4 MiB ceiling, feeding one
    # consumer. Without spill the run stalls once all three complete; with it, the heavy held
    # results are written to disk to make room, then read back to feed the consumer correctly.
    cap = 4 * _SMALL
    producers = [_producer("p1", 100), _producer("p2", 20), _producer("p3", 3)]
    consumer = Job(key="sum", run=passthrough, inputs={"add": 0},
                   paths=(_cpu(),), dependencies=("p1", "p2", "p3"))
    report = Scheduler(_small_machine(cap), margin=0.0,
                       spill_floor_bytes=_SMALL).run(JobGraph(producers + [consumer]))
    assert report.results["sum"] == 123          # restored deps fed the consumer correctly
    assert report.spill_count >= 1               # the degraded path fired
    assert report.spilled_bytes >= 2 * _SMALL
    assert "sum" not in report.cancelled


def test_spill_exhausted_aborts():
    # Same overflow, but a zero disk budget: the held results fill RAM and have nowhere to spill,
    # the one true runtime dead-end. The run aborts loud rather than hanging.
    cap = 4 * _SMALL
    producers = [_producer("p1", 1), _producer("p2", 2), _producer("p3", 3)]
    consumer = Job(key="sum", run=passthrough, inputs={"add": 0},
                   paths=(_cpu(),), dependencies=("p1", "p2", "p3"))
    sched = Scheduler(_small_machine(cap), margin=0.0, spill_floor_bytes=_SMALL,
                      spill_budget_bytes=0)
    with pytest.raises(TetradromeError, match="stalled"):
        sched.run(JobGraph(producers + [consumer]))


def consume_big(inputs, deps):
    return len(inputs["blob"])


def test_input_spill_relieves_resident_input_pressure():
    # Three independent jobs whose heavy inputs (2 MiB each, 6 MiB) overflow a 4 MiB ceiling before
    # any runs. The scheduler spills the heavy resident inputs to disk to admit the jobs, then
    # reads each back to feed its job. Outputs are tiny, so the pressure is purely resident input.
    cap = 4 * _SMALL
    jobs = [Job(key=f"j{i}", run=consume_big, inputs={"blob": bytes(2 * _SMALL)},
                paths=(_cpu(),)) for i in range(3)]
    report = Scheduler(_small_machine(cap), margin=0.0,
                       spill_floor_bytes=_SMALL).run(JobGraph(jobs))
    assert all(report.results[f"j{i}"] == 2 * _SMALL for i in range(3))  # restored inputs intact
    assert report.spill_count >= 1                                       # input spill fired


def split_three(inputs, deps):
    return {"a": 10, "b": 20, "c": 30}


def echo_shard(inputs, deps):
    (only,) = deps.values()             # a consumer of one shard sees exactly that shard
    return only * 2


def split_wrong(inputs, deps):
    return {"a": 1}                     # declares a, b but returns only a -- a contract violation


def test_partitioned_result_routes_each_shard_to_its_consumer():
    # A partitioned producer returns one payload per declared shard; each consumer depends on a
    # single Shard and receives only that shard, never the whole output. The producer holds no
    # whole result, and every shard is freed as its consumer dispatches.
    producer = Job(key="src", run=split_three, inputs={}, paths=(_cpu(),),
                   shards=frozenset({"a", "b", "c"}))
    consumers = [Job(key=f"use_{s}", run=echo_shard, inputs={}, paths=(_cpu(),),
                     dependencies={Shard("src", s)}) for s in ("a", "b", "c")]
    report = Scheduler(_machine()).run(JobGraph([producer, *consumers]))
    assert report.failures == []
    assert report.results["use_a"] == 20
    assert report.results["use_b"] == 40
    assert report.results["use_c"] == 60
    assert "src" not in report.results          # partitioned: no whole result, all shards consumed


def test_shard_dependency_on_undeclared_shard_is_rejected():
    producer = Job(key="src", run=split_three, inputs={}, paths=(_cpu(),),
                   shards=frozenset({"a", "b"}))
    bad = Job(key="use_z", run=echo_shard, inputs={}, paths=(_cpu(),),
              dependencies={Shard("src", "z")})         # z is not a declared shard
    with pytest.raises(ValueError):
        JobGraph([producer, bad])


def test_partitioned_producer_returning_wrong_shards_fails_loud():
    producer = Job(key="src", run=split_wrong, inputs={}, paths=(_cpu(),),
                   shards=frozenset({"a", "b"}))
    consumer = Job(key="use_a", run=echo_shard, inputs={}, paths=(_cpu(),),
                   dependencies={Shard("src", "a")})
    with pytest.raises(TetradromeError):
        Scheduler(_machine()).run(JobGraph([producer, consumer]))


def big_blob(inputs, deps):
    return bytes(2 * _SMALL)            # a large whole result many consumers will each want


def blob_len(inputs, deps):
    (only,) = deps.values()             # a consumer reads the whole shared result
    return len(only)


def test_shared_residence_materializes_one_segment_for_many_consumers():
    # A large result with several consumers is materialized once into a shared-memory segment the
    # consumers map, not pickled into each worker. Every consumer reads it intact, and exactly one
    # segment is made regardless of fan-out.
    producer = Job(key="src", run=big_blob, inputs={}, paths=(_cpu(),))
    consumers = [Job(key=f"c{i}", run=blob_len, inputs={}, paths=(_cpu(),),
                     dependencies={"src"}) for i in range(3)]
    report = Scheduler(_machine(), shared_min_consumers=2,
                       shared_floor_bytes=_SMALL).run(JobGraph([producer, *consumers]))
    assert report.failures == []
    assert all(report.results[f"c{i}"] == 2 * _SMALL for i in range(3))   # each read the blob
    assert report.shared_count == 1                                       # one segment, not 3 copies
    assert report.shared_bytes >= 2 * _SMALL
    assert "src" not in report.results


def test_low_fanout_result_stays_private():
    # Below the consumer threshold, a result is a private copy -- no segment.
    producer = Job(key="src", run=big_blob, inputs={}, paths=(_cpu(),))
    consumer = Job(key="c0", run=blob_len, inputs={}, paths=(_cpu(),), dependencies={"src"})
    report = Scheduler(_machine(), shared_min_consumers=2,
                       shared_floor_bytes=_SMALL).run(JobGraph([producer, consumer]))
    assert report.results["c0"] == 2 * _SMALL
    assert report.shared_count == 0
