# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Integration tests for the executor: real spawn workers on a machine built from real cores.

The job callables are module-level so they pickle to spawned workers. DAG ordering is forced by
dependencies, so these are deterministic regardless of how many cores the host actually has.
"""
import os
import pickle
import signal
import time

import pytest

from tetradrome.errors import TetradromeError
from tetradrome.scheduler import (
    Calibration,
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
    scoped,
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
    Held, WarmWorker, WorkerPool, _augment_for_admission, _fetch, _overhead_probe_main,
    _start_context)
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
            worker.dispatch(0, key, report_warm_state, {}, {}, None, None, key)
        results = {}
        for _ in range(3):
            _tag, key, status, payload, _seconds = result_queue.get(timeout=30)
            assert status == "ok"
            results[key] = _fetch(payload)
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
        worker.dispatch(0, "bad", boom, {}, {}, None, None, "bad")
        worker.dispatch(0, "good", produce, {"value": 7}, {}, None, None, "good")
        outcomes = {}
        for _ in range(2):
            _tag, key, status, payload, _seconds = result_queue.get(timeout=30)
            outcomes[key] = (status, _fetch(payload) if status == "ok" else payload)
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


def test_producers_of_a_second_calculation_wait_while_the_first_needs_the_room():
    # Two calculations, each three producers (a 2 MiB held result apiece) feeding a consumer that
    # needs 6 MiB, under a 14 MiB ceiling with nothing allowed to spill. Run greedily, all six
    # producers finish first, hold 12 MiB, and neither consumer ever fits. Held back so that a
    # started calculation's largest job still fits, the second one's producers wait and both
    # finish.
    cap = 14 * _SMALL
    jobs = []
    for name, base in (("a", 1), ("b", 10)):
        producers = [_producer((name, i), base * (i + 1)) for i in range(3)]
        consumer = Job(key=(name, "sum"), run=passthrough, inputs={"add": 0},
                       paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=6 * _SMALL),),
                       dependencies=[producer.key for producer in producers])
        jobs += producers + [consumer]
    report = Scheduler(_small_machine(cap), margin=0.0, spill_floor_bytes=_SMALL,
                       spill_budget_bytes=0).run(JobGraph(jobs))
    assert report.failures == []
    assert report.results == {("a", "sum"): 6, ("b", "sum"): 60}
    assert report.spill_count == 0


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


def test_partitioned_producer_returning_wrong_shards_fails_its_component():
    producer = Job(key="src", run=split_wrong, inputs={}, paths=(_cpu(),),
                   shards=frozenset({"a", "b"}))
    consumer = Job(key="use_a", run=echo_shard, inputs={}, paths=(_cpu(),),
                   dependencies={Shard("src", "a")})
    report = Scheduler(_machine()).run(JobGraph([producer, consumer]))
    (_, failed_key, text), = report.failures
    assert failed_key == "src"
    assert "must return exactly its declared shards" in text
    assert report.cancelled == {"src", "use_a"}


def big_blob(inputs, deps):
    return bytes(2 * _SMALL)            # a large whole result many consumers will each want


def blob_len(inputs, deps):
    (only,) = deps.values()             # a consumer reads the whole shared result
    return len(only)


def test_large_result_is_stored_once_by_its_worker_and_read_by_every_consumer(tmp_path):
    # A result too large for its completion message is written to the result store by the worker
    # that computed it: once, however many jobs consume it. Each consumer reads it there.
    producer = Job(key="src", run=big_blob, inputs={}, paths=(_cpu(),))
    consumers = [Job(key=f"c{i}", run=blob_len, inputs={}, paths=(_cpu(),),
                     dependencies={"src"}) for i in range(3)]
    report = Scheduler(_machine(), result_dir=str(tmp_path)).run(JobGraph([producer, *consumers]))
    assert report.failures == []
    assert all(report.results[f"c{i}"] == 2 * _SMALL for i in range(3))
    assert report.stored_count == 1
    assert report.stored_bytes == len(pickle.dumps(bytes(2 * _SMALL), pickle.HIGHEST_PROTOCOL))
    assert "src" not in report.results


def test_small_result_rides_in_its_completion_message(tmp_path):
    jobs = [_job("gen", produce, {"value": 5}),
            _job("end", passthrough, {"add": 1}, deps=("gen",))]
    report = Scheduler(_machine(), result_dir=str(tmp_path)).run(JobGraph(jobs))
    assert report.results == {"end": 6}
    assert report.stored_count == 0


def value_and_stored_files(inputs, deps):
    (value,) = deps.values()
    return value, sorted(name for _, _, names in os.walk(inputs["root"]) for name in names)


def test_a_stored_result_with_one_consumer_is_deleted_by_that_consumer(tmp_path):
    # "src" is stored, read by "len" alone, and so handed over to it. By the time "after" runs,
    # the store holds nothing.
    jobs = [
        Job(key="src", run=big_blob, inputs={}, paths=(_cpu(),)),
        Job(key="len", run=blob_len, inputs={}, paths=(_cpu(),), dependencies={"src"}),
        Job(key="after", run=value_and_stored_files, inputs={"root": str(tmp_path)},
            paths=(_cpu(),), dependencies={"len"}),
    ]
    report = Scheduler(_machine(), result_dir=str(tmp_path)).run(JobGraph(jobs))
    assert report.failures == []
    assert report.stored_count == 1
    assert report.results["after"] == (2 * _SMALL, [])


def test_the_result_store_is_removed_when_the_run_ends(tmp_path):
    # A stored terminal result is read into the report, and the store goes with the run.
    report = Scheduler(_machine(), result_dir=str(tmp_path)).run(
        JobGraph([Job(key="src", run=big_blob, inputs={}, paths=(_cpu(),))]))
    assert report.results["src"] == bytes(2 * _SMALL)
    assert report.stored_count == 1
    assert os.listdir(tmp_path) == []


def test_a_store_left_by_a_dead_process_is_removed_and_a_live_one_kept(tmp_path):
    process = _start_context().Process(target=time.sleep, args=(0,))
    process.start()
    process.join()
    abandoned = tmp_path / f"tetradrome-results-{process.pid}-left"
    abandoned.mkdir()
    (abandoned / "00000001").write_bytes(b"x")
    mine = tmp_path / f"tetradrome-results-{os.getpid()}-other-run"
    mine.mkdir()
    Scheduler(_machine(), result_dir=str(tmp_path)).run(
        JobGraph([Job(key="src", run=big_blob, inputs={}, paths=(_cpu(),))]))
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [mine.name]


# -- device-resident results (fabricated device buffer, plain warm worker stands in for CUDA) --

class FakeDeviceBuffer:
    """Stands in for a cupy array held in the warm worker's context: it has nbytes and carries the
    payload. It is created inside the warm worker and never crosses a process boundary."""
    def __init__(self, data):
        self.data = data
        self.nbytes = len(data)


def make_device_buffer(inputs, deps):
    return FakeDeviceBuffer(b"resident-payload")


def read_device_buffer(inputs, deps):
    (buf,) = deps.values()                  # resolved from the worker's device registry
    return buf.data


def test_device_resident_result_stays_in_worker_and_consumer_reads_it():
    # The producer's buffer stays in the warm worker's VRAM (only a handle returns), and a warm
    # consumer reads it in place -- the result never comes to host, and it is counted and sized.
    producer = Job(key="src", run=make_device_buffer, inputs={}, paths=(_gpu(_GIB // 10),),
                   cost=1000, device_resident=True)
    consumer = Job(key="c", run=read_device_buffer, inputs={}, paths=(_gpu(_GIB // 10),),
                   cost=1000, dependencies={"src"})
    report = Scheduler(_gpu_machine(), warm_setup=warm_setup, warm_between=warm_between,
                       context_vram_reserve=0).run(JobGraph([producer, consumer]))
    assert report.failures == []
    assert report.results["c"] == b"resident-payload"   # consumer read the resident buffer
    assert "src" not in report.results                  # the buffer never came back to host
    assert report.device_count == 1
    assert report.device_bytes == len(b"resident-payload")


def test_device_resident_producer_routed_fresh_degrades_to_host(caplog):
    # A big-vram device-resident job trips the fresh trigger. A fresh process cannot hold the
    # buffer, so the scheduler warns and produces a host-resident result instead of aborting.
    big = Job(key="src", run=make_device_buffer, inputs={}, paths=(_gpu(int(_GIB * 0.6)),),
              device_resident=True)
    with caplog.at_level("WARNING"):
        report = Scheduler(_gpu_machine(), warm_setup=warm_setup, warm_between=warm_between,
                           context_vram_reserve=0).run(JobGraph([big]))
    assert report.failures == []
    assert report.device_count == 0                         # never made it onto the device
    assert report.results["src"].data == b"resident-payload"   # came back to the host instead
    assert any("device residency skipped" in r.message for r in caplog.records)


def test_fresh_gpu_consumer_of_device_resident_result_is_run_warm(caplog):
    # The producer keeps its result on-device; a GPU consumer that would route fresh is run warm
    # instead so it can read the buffer in-context. It still reads the resident result, with a
    # warning that it was serialized.
    producer = Job(key="src", run=make_device_buffer, inputs={}, paths=(_gpu(_GIB // 10),),
                   cost=1000, device_resident=True)
    fresh_consumer = Job(key="c", run=read_device_buffer, inputs={},
                         paths=(_gpu(int(_GIB * 0.6)),), dependencies={"src"})
    with caplog.at_level("WARNING"):
        report = Scheduler(_gpu_machine(), warm_setup=warm_setup, warm_between=warm_between,
                           context_vram_reserve=0).run(JobGraph([producer, fresh_consumer]))
    assert report.failures == []
    assert report.device_count == 1                         # producer's buffer stayed resident
    assert report.results["c"] == b"resident-payload"       # consumer read it after being run warm
    assert any("running it warm" in r.message for r in caplog.records)


def test_cpu_consumer_downgrades_device_resident_result_to_host(caplog):
    # A device-resident producer with a consumer that can run on the CPU cannot keep its result on
    # the device (the CPU consumer could not read it), so it warns and produces a host result the
    # consumer reads normally.
    producer = Job(key="src", run=make_device_buffer, inputs={}, paths=(_gpu(_GIB // 10),),
                   cost=1000, device_resident=True)
    cpu_consumer = Job(key="c", run=read_device_buffer, inputs={}, paths=(_cpu(),),
                       dependencies={"src"})
    with caplog.at_level("WARNING"):
        report = Scheduler(_gpu_machine(), warm_setup=warm_setup, warm_between=warm_between,
                           context_vram_reserve=0).run(JobGraph([producer, cpu_consumer]))
    assert report.failures == []
    assert report.device_count == 0
    assert report.results["c"] == b"resident-payload"       # CPU consumer read the host result
    assert any("not GPU-only" in r.message for r in caplog.records)


# -- the inline path: called in the scheduler's own process --

def _inline():
    return ComputePath(Placement.INLINE, cores=1, ram_bytes=_SMALL)


def whoami(inputs, deps):
    return os.getpid()


def test_inline_job_runs_in_the_scheduler_process():
    jobs = [
        Job(key="here", run=whoami, inputs={}, paths=(_inline(),)),
        _job("there", whoami, {}),
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.failures == []
    assert report.results["here"] == os.getpid()
    assert report.results["there"] != os.getpid()


def test_inline_job_sits_between_workers():
    jobs = [
        _job("gen", produce, {"value": 10}),
        Job(key="mid", run=passthrough, inputs={"add": 5}, paths=(_inline(),),
            dependencies=("gen",)),                                # 10 + 5, in this process
        _job("end", passthrough, {"add": 1}, deps=("mid",)),       # 15 + 1
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.results["end"] == 16
    assert report.timings.keys() == {"gen", "mid", "end"}


def test_inline_failure_abandons_its_component_only():
    jobs = [
        _job("genX", produce, {"value": 1}),
        Job(key="boom", run=boom, inputs={}, paths=(_inline(),), dependencies=("genX",)),
        _job("endX", passthrough, {"add": 0}, deps=("boom",)),
        _job("genY", produce, {"value": 7}),
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.results.get("genY") == 7
    assert {"genX", "boom", "endX"} <= report.cancelled
    (_, failed_key, text), = report.failures
    assert failed_key == "boom"
    assert "kaboom" in text


def test_inline_job_reads_a_stored_result():
    # The producer's result is stored by its worker; the inline consumer reads it from the store
    # as a worker would.
    jobs = [
        Job(key="src", run=big_blob, inputs={}, paths=(_cpu(),)),
        Job(key="c0", run=blob_len, inputs={}, paths=(_inline(),), dependencies={"src"}),
        Job(key="c1", run=blob_len, inputs={}, paths=(_cpu(),), dependencies={"src"}),
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.failures == []
    assert report.stored_count == 1
    assert report.results["c0"] == 2 * _SMALL
    assert report.results["c1"] == 2 * _SMALL


# -- the planner: measured rates and dispatch overheads choose the path --

def _either(key, kind, cost):
    return Job(key=key, run=whoami, inputs={}, kind=kind, cost=cost,
               paths=(ComputePath(Placement.INLINE, cores=1, ram_bytes=_SMALL), _cpu()))


def test_a_run_measures_the_dispatch_overhead_of_its_workers():
    report = Scheduler(_machine()).run(JobGraph([_job("a", produce, {"value": 1})]))
    overhead = report.calibration.overhead(Placement.CPU_PINNED)
    assert overhead is not None and overhead > 0.0
    assert report.calibration.overhead(Placement.CPU_UNPINNED) is None


def test_calibrated_scheduler_sends_long_jobs_to_workers_and_short_ones_inline():
    # One unit of "probe" work is known to take a second; a worker costs 50 ms to dispatch. A
    # job of 0.01 units is over before a worker would start, so it runs here; a job of 10 units
    # is not, so it gets a worker though both list the inline path first.
    calibration = Calibration()
    calibration.observe(cost=1.0, placement=Placement.CPU_PINNED, seconds=1.0, kind="probe")
    calibration.observe_overhead(Placement.CPU_PINNED, 0.05)
    jobs = [_either("short", "probe", 0.01), _either("long", "probe", 10.0)]
    report = Scheduler(_machine(), calibration=calibration).run(JobGraph(jobs))
    assert report.failures == []
    assert report.results["short"] == os.getpid()
    assert report.results["long"] != os.getpid()


def test_uncalibrated_scheduler_follows_the_declared_order():
    report = Scheduler(_machine()).run(JobGraph([_either("first", "probe", 10.0)]))
    assert report.results["first"] == os.getpid()


def test_a_shared_calibration_carries_what_one_run_measured_into_the_next():
    calibration = Calibration()
    scheduler = Scheduler(_machine(), calibration=calibration)
    scheduler.run(JobGraph([Job(key="warm-up", run=nap, inputs={"seconds": 0.3}, kind="probe",
                                cost=1.0, paths=(_cpu(),))]))
    assert calibration.rate(Placement.CPU_PINNED, "probe") >= 0.3
    # The first run measured about 0.3 s per unit and a worker's dispatch cost, so 100 units is
    # known to outlast a dispatch and the next run sends it to a worker.
    report = scheduler.run(JobGraph([_either("next", "probe", 100.0)]))
    assert report.results["next"] != os.getpid()


# -- a worker that dies without reporting --

def vanish(inputs, deps):
    os._exit(inputs["code"])


def killed(inputs, deps):
    os.kill(os.getpid(), signal.SIGKILL)


def test_worker_that_exits_without_reporting_fails_its_component():
    jobs = [
        _job("dead", vanish, {"code": 3}),
        _job("after", passthrough, {"add": 0}, deps=("dead",)),
        _job("other", produce, {"value": 7}),
    ]
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.results == {"other": 7}
    (component, failed_key, text), = report.failures
    assert failed_key == "dead"
    assert component == {"dead", "after"}
    assert "exited with code 3 without reporting a result" in text


def test_worker_killed_by_a_signal_is_reported_not_waited_for():
    # What the kernel does to a process that runs the box out of memory.
    report = Scheduler(_machine()).run(JobGraph([_job("gone", killed, {})]))
    (_, failed_key, text), = report.failures
    assert failed_key == "gone"
    assert f"exited with code {-signal.SIGKILL} without reporting a result" in text


# -- several copies of one graph shape in one graph --

def _chain(value):
    return [
        _job("gen", produce, {"value": value}),
        _job("end", passthrough, {"add": 1}, deps=("gen",)),
    ]


def test_scoped_jobs_are_renamed_with_their_dependencies():
    plain, split = _chain(1)
    sharded = Job(key="use", run=produce, inputs={}, paths=(_cpu(),),
                  dependencies={Shard("gen", "half")})
    renamed = scoped([plain, split, sharded], "k")
    assert [job.key for job in renamed] == [("k", "gen"), ("k", "end"), ("k", "use")]
    assert renamed[0].dependencies == frozenset()
    assert renamed[1].dependencies == {("k", "gen")}
    assert renamed[2].dependencies == {Shard(("k", "gen"), "half")}


def test_scoped_job_sees_its_dependencies_under_their_own_keys():
    job = scoped([_job("end", passthrough, {"add": 1}, deps=("gen",))], "k")[0]
    assert job.run({"add": 1}, {("k", "gen"): 10}) == 11


def test_two_copies_of_a_graph_run_side_by_side():
    jobs = scoped(_chain(10), "a") + scoped(_chain(20), "b")
    report = Scheduler(_machine()).run(JobGraph(jobs))
    assert report.failures == []
    assert report.results == {("a", "end"): 11, ("b", "end"): 21}


def started_at(inputs, deps):
    return time.monotonic()


def test_ready_jobs_are_tried_in_graph_order():
    # One core. After "first" finishes, its consumer and the unrelated "other" are both ready; the
    # consumer comes earlier in the graph, so it runs first even though it became ready later.
    core = frozenset({min(os.sched_getaffinity(0))})
    machine = Machine(nodes=(NumaNode(0, core, 8 * _GIB),), gpus=(), mem_cap_bytes=8 * _GIB)
    jobs = [
        _job("first", produce, {"value": 1}),
        _job("consumer", started_at, {}, deps=("first",)),
        _job("other", started_at, {}),
    ]
    report = Scheduler(machine).run(JobGraph(jobs))
    assert report.failures == []
    assert report.results["consumer"] < report.results["other"]


def test_among_ready_jobs_the_longest_predicted_chain_starts_first():
    # One core. "feeds" is the shortest job itself but the longest hangs on it, so its chain is
    # the longest and it runs first; "fed" follows, then the longer of the other two. In graph
    # order "fed" would have started last.
    core = frozenset({min(os.sched_getaffinity(0))})
    machine = Machine(nodes=(NumaNode(0, core, 8 * _GIB),), gpus=(), mem_cap_bytes=8 * _GIB)
    calibration = Calibration()
    calibration.fix({"rates": [{"kind": "probe", "device": "cpu", "seconds_fixed": 0.0,
                                "seconds_per_unit": 1.0}],
                     "overheads": [], "sibling": []})

    def probe(key, cost, deps=()):
        return Job(key=key, run=started_at, inputs={}, kind="probe", cost=cost,
                   paths=(_cpu(),), dependencies=deps)

    jobs = [probe("short", 2.0), probe("long", 5.0), probe("feeds", 1.0),
            probe("fed", 9.0, deps=("feeds",))]
    report = Scheduler(machine, calibration=calibration).run(JobGraph(jobs))
    assert report.failures == []
    started = report.results
    assert started["fed"] < started["long"] < started["short"]


def pids_so_far(inputs, deps):
    return [pid for pids in deps.values() for pid in pids] + [os.getpid()]


def _pid_chain(length):
    return [Job(key=index, run=pids_so_far, inputs={}, paths=(_cpu(),),
                dependencies=[index - 1] if index else [])
            for index in range(length)]


def test_a_run_reuses_its_workers():
    # Three jobs one after another: one worker process runs them all.
    report = Scheduler(_machine()).run(JobGraph(_pid_chain(3)))
    assert report.failures == []
    first, second, third = report.results[2]
    assert first == second == third != os.getpid()


def test_a_worker_left_holding_too_much_memory_is_retired():
    # With nothing allowed to be left behind, every job retires its worker and the next job
    # gets a new one.
    report = Scheduler(_machine(), pool_retire_bytes=0).run(JobGraph(_pid_chain(3)))
    assert report.failures == []
    assert len(set(report.results[2])) == 3


def test_a_job_after_a_killed_worker_gets_another_worker():
    # "die" kills its worker and its component fails. The unrelated job after it still runs.
    jobs = [Job(key="die", run=killed, inputs={}, paths=(_cpu(),)),
            Job(key="after", run=produce, inputs={"value": 4}, paths=(_cpu(),))]
    core = frozenset({min(os.sched_getaffinity(0))})
    machine = Machine(nodes=(NumaNode(0, core, 8 * _GIB),), gpus=(), mem_cap_bytes=8 * _GIB)
    report = Scheduler(machine).run(JobGraph(jobs))
    assert [failed for _, failed, _ in report.failures] == ["die"]
    assert report.results == {"after": 4}


def test_a_kept_pool_hands_its_workers_to_the_next_run():
    pool = WorkerPool()
    try:
        first = Scheduler(_machine(), pool=pool).run(JobGraph(_pid_chain(2))).results[1]
        second = Scheduler(_machine(), pool=pool).run(JobGraph(_pid_chain(2))).results[1]
    finally:
        pool.close()
    assert len(set(first)) == 1
    assert set(second) == set(first)


def test_a_result_posted_for_an_earlier_run_is_ignored():
    # A job stopped when a run ends can still post its result. The next run, here reusing the
    # key, must not take it for its own.
    pool = WorkerPool()
    try:
        Scheduler(_machine(), pool=pool).run(JobGraph([_job("x", produce, {"value": 1})]))
        stale = Held(0, data=pickle.dumps("stale"))
        pool.results().put((1, "x", "ok", stale, 0.0, 0))        # the first run's tag was 1
        report = Scheduler(_machine(), pool=pool).run(
            JobGraph([_job("x", produce, {"value": 2})]))
    finally:
        pool.close()
    assert report.results == {"x": 2}


def test_idle_workers_are_trimmed_to_a_memory_budget():
    pool = WorkerPool()
    try:
        Scheduler(_machine(), pool=pool).run(JobGraph(_pid_chain(2)))
        assert len(pool.idle) == 1
        pool.trim(0)
        assert pool.idle == [] and pool.workers == []
    finally:
        pool.close()


def test_report_carries_each_worker_jobs_dispatch_overhead():
    report = Scheduler(_machine()).run(JobGraph(_pid_chain(3)))
    assert sorted(report.overheads) == [0, 1, 2]
    assert all(seconds >= 0 for seconds in report.overheads.values())


def test_on_complete_is_called_once_per_finished_job():
    seen = []
    Scheduler(_machine(), on_complete=seen.append).run(JobGraph(_chain(1)))
    assert sorted(seen) == ["end", "gen"]
