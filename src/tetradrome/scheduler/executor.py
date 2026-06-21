# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The executor: run a JobGraph to completion, placing each job on the machine.

Workers are ephemeral processes under a spawn context, so every job starts from clean state and
frees all its memory at exit; the only memory in play is the jobs running right now. Each worker
pins itself to its granted cores (and, for a GPU job, to its granted device) before running. A
daemon thread samples each worker's resident memory and posts it to the loop, which charges
max(declared, actual) so admission self-corrects against under-prediction; a job that crosses
its declared peak is logged once, and its peak versus its prediction is logged at completion so
the cost model can be tuned.

Failure is isolated to the lineage. When a worker raises, the whole connected component it
belongs to (one knot's DAG) is abandoned: its unstarted jobs are cancelled and its running
siblings are killed, since their results would feed an assembly that can no longer complete,
and their allocations return to the ledger. Every other component keeps running. The run
finishes and reports which jobs produced results and which components failed and why.

The loop never reserves or forces an order: among ready jobs it admits whatever fits now, so the
machine stays as full as the work allows, and a job that cannot be placed yet simply waits.
"""
from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import os
import queue
import sys
import threading
import time
import traceback

from ..errors import TetradromeError
from .graph import JobGraph
from .inventory import Machine
from .job import Placement
from .ledger import Allocation, Ledger
from .placement import Outcome, plan_placement
from .routing import Calibration, Execution, route_execution

logger = logging.getLogger(__name__)

_GRACE_SECONDS = 2.0

# Modules the forkserver imports once so every forked worker inherits them warm, turning a
# cold ~400ms import per job into a ~1ms fork. Best-effort: a module that cannot import (e.g.
# numpy absent) is skipped and that worker just imports it cold, so the pure-Python floor still
# runs. CUDA is deliberately not warmed here -- a context cannot survive a fork, so a GPU worker
# initializes its device after forking.
_FORKSERVER_PRELOAD = [
    "numpy",
    "tetradrome.algebra",
    "tetradrome.engines.floer.scheduling",
    "tetradrome.scheduler.executor",
]


def _start_context():
    """The process-start mechanism best suited to the platform: forkserver where the OS can
    fork (Linux, macOS), so workers inherit a warm interpreter cheaply and still die clean;
    spawn on Windows, which cannot fork, so it pays the cold start the OS forces."""
    if sys.platform == "win32":
        return multiprocessing.get_context("spawn")
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(_FORKSERVER_PRELOAD)
    return context


def _worker_main(key, run, inputs, deps, cores, gpu_index, result_queue):
    """Top-level worker entry, picklable for spawn: pin, time the run, report one message.

    The reported time covers only ``run`` itself, not affinity or device setup, so it is the
    work the cost model predicts. A failure reports no time (0.0): a job that did not finish
    carries no usable runtime, and including partial times would poison the calibration.
    """
    try:
        os.sched_setaffinity(0, set(cores))
        if gpu_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""   # a CPU-placed worker must not grab a GPU
        t0 = time.perf_counter()
        result = run(inputs, deps)
        compute_time = time.perf_counter() - t0
    except Exception:
        result_queue.put((key, "error", traceback.format_exc(), 0.0))
        return
    result_queue.put((key, "ok", result, compute_time))


def _warm_worker_main(cores, gpu_index, job_queue, result_queue, setup, between):
    """A persistent worker: pin once, optionally set up a session (a held CUDA context on the
    GPU), then run jobs serially off ``job_queue`` until a None sentinel, posting each result to
    the shared ``result_queue`` in the same shape an ephemeral worker uses so the loop reaps warm
    and fresh jobs identically.

    Strictly one job at a time: the held context drives a single stream, and serial execution is
    also what keeps the memory sampler honest, since only one job's footprint is ever live in this
    pid at once. ``between`` runs after every job, success or failure, to release per-job resources
    (it frees the GPU memory pool); ``setup`` runs once before the first job. This worker is only
    ever fed jobs the router classed small, so serializing them costs nothing the routing did not
    already decide to accept.
    """
    os.sched_setaffinity(0, set(cores))
    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if setup is not None:
        setup()
    while True:
        item = job_queue.get()
        if item is None:
            return
        key, run, inputs, deps = item
        try:
            t0 = time.perf_counter()
            result = run(inputs, deps)
            result_queue.put((key, "ok", result, time.perf_counter() - t0))
        except Exception:
            result_queue.put((key, "error", traceback.format_exc(), 0.0))
        finally:
            if between is not None:
                between()


class WarmWorker:
    """A handle to one persistent serial worker, one per GPU, that holds a session across jobs.

    Started once, fed jobs one at a time with ``dispatch``, and stopped with ``shutdown`` at run
    end. Results land on the shared result queue, not here, so the executor's reap loop is the
    single place that handles every job's outcome. ``setup`` and ``between`` are module-level
    callables (picklable for the worker process); the GPU build supplies ones that create a CUDA
    context and free its memory pool between jobs, and the default None pair makes a plain serial
    worker for exercising dispatch and lifecycle without a device.
    """

    def __init__(self, ctx, cores, gpu_index, result_queue, setup=None, between=None):
        self._job_queue = ctx.Queue()
        self._proc = ctx.Process(target=_warm_worker_main, args=(
            cores, gpu_index, self._job_queue, result_queue, setup, between))
        self._proc.start()

    @property
    def pid(self):
        return self._proc.pid

    def dispatch(self, key, run, inputs, deps) -> None:
        self._job_queue.put((key, run, inputs, deps))

    def shutdown(self) -> None:
        self._job_queue.put(None)               # let the worker finish the current job and exit
        self._proc.join(timeout=_GRACE_SECONDS)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join()


def _read_private_bytes(pid: int) -> int | None:
    """Private resident bytes for a pid (USS = Private_Clean + Private_Dirty from smaps_rollup),
    or None if the process is gone or the file is unavailable.

    Under forkserver every worker shares the parent's warm pages (numpy, the package) copy-on-
    write, so RSS would count that inherited footprint in full in every worker and summing it
    would phantom-charge memory that physically exists once. Private bytes are what the job
    actually added on top of the shared base, which is the marginal footprint the ledger should
    charge and the right thing to compare against the cost model's predicted peak.
    """
    private = 0
    try:
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                if line.startswith("Private_Clean:") or line.startswith("Private_Dirty:"):
                    private += int(line.split()[1]) * 1024   # the field is in kB
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None
    return private


class _Sampler(threading.Thread):
    """A pure reader: snapshot the running pids, read each one's private memory, post to the
    loop's queue."""

    def __init__(self, snapshot, sink: queue.Queue, interval: float, stop: threading.Event):
        super().__init__(daemon=True)
        self._snapshot = snapshot
        self._sink = sink
        self._interval = interval
        self._stopping = stop

    def run(self) -> None:
        while not self._stopping.is_set():
            for key, pid in self._snapshot().items():
                private = _read_private_bytes(pid)
                if private is not None:
                    self._sink.put((key, private))
            self._stopping.wait(self._interval)


@dataclasses.dataclass
class RunReport:
    """The outcome of a run: results for completed jobs, and one entry per failed component.

    ``timings`` maps each completed job's key to the seconds its ``run`` took, the measured work
    behind the cost model: comparing it to the job's predicted ``cost`` is what calibrates the
    per-op constant and, in turn, the warm-versus-fresh routing.
    """
    results: dict
    failures: list          # (component frozenset, failed_key, error_text)
    cancelled: frozenset
    timings: dict = dataclasses.field(default_factory=dict)
    calibration: "Calibration | None" = None     # final per-placement rates, for inspection


class Scheduler:
    """Runs a JobGraph on a machine, ephemeral spawn workers, no forced order."""

    def __init__(self, machine: Machine, margin: float = 0.03,
                 sample_interval: float = 0.5, numba_cache_dir: str | None = None,
                 context_overhead: float = float("inf"), vram_fraction: float = 0.5,
                 time_multiple: float = 10.0, context_vram_reserve: int = 0,
                 warm_setup=None, warm_between=None):
        self.machine = machine
        self.margin = margin
        self.sample_interval = sample_interval
        self.numba_cache_dir = numba_cache_dir
        # Warm-versus-fresh routing knobs. context_overhead defaults to infinity so the time
        # trigger stays off until it is measured on a real device; until then the decision is
        # vram-only. context_vram_reserve is the held context's baseline VRAM, held back from a
        # GPU's budget while its warm worker is up. warm_setup/warm_between are the warm worker's
        # session hooks; the GPU build supplies the pair that holds a CUDA context and frees its
        # pool, and None makes a plain serial worker.
        self.context_overhead = context_overhead
        self.vram_fraction = vram_fraction
        self.time_multiple = time_multiple
        self.context_vram_reserve = context_vram_reserve
        self.warm_setup = warm_setup
        self.warm_between = warm_between

    def run(self, graph: JobGraph) -> RunReport:
        if self.numba_cache_dir:
            os.environ["NUMBA_CACHE_DIR"] = self.numba_cache_dir
        ctx = _start_context()
        result_queue = ctx.Queue()
        sample_queue: queue.Queue = queue.Queue()
        ledger = Ledger(self.machine)

        running: dict = {}              # key -> Process, fresh jobs with their own process
        warm_running: dict = {}         # key -> gpu_index, jobs dispatched to a warm worker
        warm_workers: dict = {}         # gpu_index -> WarmWorker, started lazily
        running_pids: dict = {}         # key -> pid, shared with the sampler under the lock
        pids_lock = threading.Lock()
        declared_ram: dict = {}         # key -> declared peak, for warnings and the summary
        peak_actual: dict = {}          # key -> max sampled private bytes
        over_warned: set = set()        # keys already warned for crossing their declaration
        placement_of: dict = {}         # key -> the placement it ran on, for calibration
        calibration = Calibration()
        results: dict = {}
        timings: dict = {}              # key -> measured run seconds, for cost calibration
        completed: set = set()
        cancelled: set = set()
        failures: list = []

        def is_running(key) -> bool:
            return key in running or key in warm_running

        def snapshot() -> dict:
            with pids_lock:
                return dict(running_pids)

        def ensure_warm(gpu_index):
            worker = warm_workers.get(gpu_index)
            if worker is not None:
                return worker
            gpu = next(g for g in self.machine.gpus if g.index == gpu_index)
            node = gpu.numa_node if gpu.numa_node is not None else self.machine.nodes[0].index
            cores = next((n.cores for n in self.machine.nodes if n.index == node),
                         self.machine.nodes[0].cores)
            worker = WarmWorker(ctx, cores, gpu_index, result_queue,
                                self.warm_setup, self.warm_between)
            warm_workers[gpu_index] = worker
            if self.context_vram_reserve > 0:
                # Hold the held context's baseline VRAM out of the budget so admission cannot
                # hand out device memory the context is already using.
                ledger.add(Allocation(
                    job_key=("__warm_context__", gpu_index), placement=Placement.GPU, cores=0,
                    declared_ram=0, node_index=node, gpu_index=gpu_index,
                    declared_vram=self.context_vram_reserve))
            return worker

        def route_warm(job, placed) -> bool:
            gpu = next(g for g in self.machine.gpus if g.index == placed.gpu_index)
            predicted = calibration.predicted_time(float(job.cost), Placement.GPU)
            decision = route_execution(
                predicted_vram=placed.path.vram_bytes, vram_budget=gpu.vram_bytes,
                predicted_time=predicted, context_overhead=self.context_overhead,
                vram_fraction=self.vram_fraction, time_multiple=self.time_multiple)
            return decision is Execution.WARM

        def launch(job, placed) -> None:
            deps = {dep: results[dep] for dep in job.dependencies}
            placement = placed.path.placement
            ledger.add(Allocation(
                job_key=job.key, placement=placement, cores=placed.cores,
                declared_ram=placed.path.ram_bytes, node_index=placed.node_index,
                gpu_index=placed.gpu_index, declared_vram=placed.path.vram_bytes,
            ))
            declared_ram[job.key] = placed.path.ram_bytes
            placement_of[job.key] = placement
            if placement is Placement.GPU and route_warm(job, placed):
                # Small GPU job: serialize it through the held context. Not RAM-sampled -- it is
                # small by the gate, and only one warm job's footprint is ever live in the shared
                # pid at a time, so a live sample could not be attributed to it cleanly anyway.
                ensure_warm(placed.gpu_index).dispatch(job.key, job.run, job.inputs, deps)
                warm_running[job.key] = placed.gpu_index
            else:
                proc = ctx.Process(target=_worker_main, args=(
                    job.key, job.run, job.inputs, deps,
                    placed.cores, placed.gpu_index, result_queue,
                ))
                proc.start()
                running[job.key] = proc
                with pids_lock:
                    running_pids[job.key] = proc.pid
            if placed.note:
                logger.info("job %r degraded: %s", job.key, placed.note)

        def forget_warm(key) -> None:
            # Drop a warm job's bookkeeping without touching the shared worker. A stray result
            # that arrives after this is ignored by reap_one's not-running guard.
            warm_running.pop(key, None)
            ledger.remove(key)

        def stop_worker(key) -> None:
            proc = running.pop(key)
            with pids_lock:
                running_pids.pop(key, None)
            ledger.remove(key)
            proc.terminate()
            proc.join(timeout=_GRACE_SECONDS)
            if proc.is_alive():
                proc.kill()
                proc.join()

        def poison(failed_key, error_text) -> None:
            component = graph.component(failed_key)
            for member in component:
                if member in running:
                    stop_worker(member)
                elif member in warm_running:
                    forget_warm(member)     # leave the shared worker up for other components
                results.pop(member, None)
                completed.discard(member)
                cancelled.add(member)
            failures.append((component, failed_key, error_text))
            logger.error("job %r failed; abandoning its component (%d job(s)): %s",
                         failed_key, len(component), error_text.strip().splitlines()[-1])

        def drain_samples() -> bool:
            got = False
            while True:
                try:
                    key, private = sample_queue.get_nowait()
                except queue.Empty:
                    break
                got = True
                if key not in running:
                    continue
                peak_actual[key] = max(peak_actual.get(key, 0), private)
                ledger.set_actual(key, private)
                if private > declared_ram[key] and key not in over_warned:
                    over_warned.add(key)
                    logger.warning("job %r exceeded predicted memory: predicted %d, private %d",
                                   key, declared_ram[key], private)
            return got

        def reap_one(item) -> None:
            key, status, payload, compute_time = item
            if not is_running(key):
                return                  # stale message from a job already reaped or killed
            if key in running:          # fresh: its own process to reap
                proc = running.pop(key)
                with pids_lock:
                    running_pids.pop(key, None)
                ledger.remove(key)
                proc.join()
                logger.info("job %r done: predicted %d, peak private %d",
                            key, declared_ram[key], peak_actual.get(key, 0))
            else:                       # warm: the worker persists, only the allocation clears
                warm_running.pop(key)
                ledger.remove(key)
                logger.info("job %r done (warm)", key)
            if status == "ok":
                results[key] = payload
                completed.add(key)
                timings[key] = compute_time
                calibration.observe(float(graph.get(key).cost), placement_of[key], compute_time)
                logger.info("job %r runtime %.4fs vs predicted cost %g",
                            key, compute_time, float(graph.get(key).cost))
            else:
                poison(key, payload)

        def reap(block) -> bool:
            got = False
            if block:
                try:
                    reap_one(result_queue.get(timeout=self.sample_interval))
                    got = True
                except queue.Empty:
                    pass
            while True:
                try:
                    reap_one(result_queue.get_nowait())
                except queue.Empty:
                    break
                got = True
            return got

        def admit() -> bool:
            did = False
            for job in graph.ready(completed):
                if is_running(job.key) or job.key in cancelled:
                    continue
                decision = plan_placement(self.machine, ledger, job, self.margin)
                if decision.outcome is Outcome.ADMIT:
                    launch(job, decision.placed)
                    did = True
                elif decision.outcome is Outcome.INFEASIBLE:
                    poison(job.key, f"infeasible placement: {decision.reason}")
                    did = True
            return did

        stop = threading.Event()
        sampler = _Sampler(snapshot, sample_queue, self.sample_interval, stop)
        sampler.start()
        total = len(graph)
        try:
            while len(completed) + len(cancelled) < total:
                progressed = drain_samples()
                progressed = reap(block=False) or progressed
                progressed = admit() or progressed
                if progressed:
                    continue
                if running or warm_running:
                    reap(block=True)
                else:
                    remaining = [job.key for job in graph.jobs()
                                 if job.key not in completed and job.key not in cancelled]
                    raise TetradromeError(
                        f"scheduler stalled with nothing running; remaining: {remaining}")
        finally:
            stop.set()
            sampler.join(timeout=1.0)
            for key in list(running):
                stop_worker(key)
            for gpu_index, worker in warm_workers.items():
                if self.context_vram_reserve > 0:
                    ledger.remove(("__warm_context__", gpu_index))
                worker.shutdown()
            result_queue.close()
            result_queue.join_thread()
        return RunReport(results=results, failures=failures,
                         cancelled=frozenset(cancelled), timings=timings,
                         calibration=calibration)
