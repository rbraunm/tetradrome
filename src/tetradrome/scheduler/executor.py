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
from .hostplatform import for_host
from .accelerator import detect_accelerator
from .inventory import Machine
from .job import Placement
from .ledger import Allocation, Ledger
from .placement import Outcome, job_feasibility, plan_placement
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


def _worker_main(key, run, inputs, deps, cores, gpu_index, result_queue, platform):
    """Top-level worker entry, picklable for spawn: pin, time the run, report one message.

    The reported time covers only ``run`` itself, not affinity or device setup, so it is the
    work the cost model predicts. A failure reports no time (0.0): a job that did not finish
    carries no usable runtime, and including partial times would poison the calibration.
    """
    try:
        platform.pin(cores)
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


def _overhead_probe_main(result_queue, platform, accelerator, session_setup):
    """One probe that measures what a worker process costs before it does any work, by being what
    the heaviest worker is. It imports the workload, and when a GPU is present it also stands up
    the device session the way a GPU worker does, so its own private RAM reflects the real
    per-worker baseline (interpreter, imports, and the context's host-side memory under spawn).

    For the context's device VRAM it reads free memory through the accelerator -- a context-free
    read -- before standing the context up and again after, so the difference is the context's
    cost rather than whatever was already resident. Reports (ram_baseline, context_vram), or None
    on any failure, so the caller fails loud rather than charging an unmeasured overhead.
    """
    import importlib
    importlib.import_module("numpy")                 # the heavy shared base a worker imports
    importlib.import_module("tetradrome.algebra")
    context_vram = 0
    try:
        if accelerator is not None and session_setup is not None:
            free_before = accelerator.free_vram_bytes()
            session_setup()                          # stand up the context, as a GPU worker does
            free_after = accelerator.free_vram_bytes()
            context_vram = max(free_before - free_after, 0)
        ram_baseline = platform.private_bytes(os.getpid())
    except Exception:
        result_queue.put(None)
        return
    if ram_baseline is None:
        result_queue.put(None)
        return
    result_queue.put((ram_baseline, context_vram))


def _augment_for_admission(paths, *, worker_shared, ram_baseline, context_vram, gpu_budget,
                           predicted_gpu_time, context_overhead, vram_fraction, time_multiple):
    """Augment a job's compute paths so admission sees each fresh worker's true footprint, not
    just its working set. A fresh worker carries two costs the path omits: a RAM baseline (the
    interpreter and imports, charged only where workers spawn rather than fork) and, for a fresh
    GPU process, the CUDA context's VRAM. Return the augmented paths plus the GPU path's
    warm-versus-fresh decision (None when the job has no GPU path or the machine has no GPU).

    A GPU path that routes warm adds nothing here: it reuses the standing warm worker, whose
    baseline and context are reserved once when it starts. CPU paths always run fresh. Charging
    these makes admission bind on memory the box actually has, so a swarm of fresh processes each
    holding an uncounted context and import baseline cannot be admitted in the first place.
    """
    fresh_ram = 0 if worker_shared else ram_baseline
    gpu_execution = None
    augmented = []
    for path in paths:
        if path.placement is Placement.GPU:
            if gpu_budget is None:
                augmented.append(path)          # no device to place it on; leave untouched
                continue
            gpu_execution = route_execution(
                predicted_vram=path.vram_bytes, vram_budget=gpu_budget,
                predicted_time=predicted_gpu_time, context_overhead=context_overhead,
                vram_fraction=vram_fraction, time_multiple=time_multiple)
            if gpu_execution is Execution.FRESH:
                augmented.append(dataclasses.replace(
                    path, ram_bytes=path.ram_bytes + fresh_ram,
                    vram_bytes=path.vram_bytes + context_vram))
            else:
                augmented.append(path)          # warm: shares the standing worker
        else:
            augmented.append(dataclasses.replace(path, ram_bytes=path.ram_bytes + fresh_ram))
    return tuple(augmented), gpu_execution


def _warm_worker_main(cores, gpu_index, job_queue, result_queue, setup, between, platform):
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
    platform.pin(cores)
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

    def __init__(self, ctx, cores, gpu_index, result_queue, setup=None, between=None,
                 platform=None):
        self._job_queue = ctx.Queue()
        self._proc = ctx.Process(target=_warm_worker_main, args=(
            cores, gpu_index, self._job_queue, result_queue, setup, between, platform))
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


class _Sampler(threading.Thread):
    """A pure reader: snapshot the running pids, read each one's private memory via the host
    platform, post to the loop's queue."""

    def __init__(self, snapshot, sink: queue.Queue, interval: float, stop: threading.Event,
                 platform):
        super().__init__(daemon=True)
        self._snapshot = snapshot
        self._sink = sink
        self._interval = interval
        self._stopping = stop
        self._platform = platform

    def run(self) -> None:
        while not self._stopping.is_set():
            for key, pid in self._snapshot().items():
                private = self._platform.private_bytes(pid)
                if private is not None:
                    self._sink.put((key, private))
            self._stopping.wait(self._interval)


@dataclasses.dataclass
class RunReport:
    """The outcome of a run: results for completed jobs, one entry per failed component, and one
    per job found infeasible up front.

    ``timings`` maps each completed job's key to the seconds its ``run`` took, the measured work
    behind the cost model: comparing it to the job's predicted ``cost`` is what calibrates the
    per-op constant and, in turn, the warm-versus-fresh routing. ``infeasible`` holds one
    InfeasibleJobError per job the bare machine could serve on no declared path; their lineages are
    abandoned but the rest of the batch runs.
    """
    results: dict
    failures: list          # (component frozenset, failed_key, error_text)
    cancelled: frozenset
    infeasible: tuple = ()  # InfeasibleJobError per job infeasible on the bare machine
    timings: dict = dataclasses.field(default_factory=dict)
    calibration: "Calibration | None" = None     # final per-placement rates, for inspection


class Scheduler:
    """Runs a JobGraph on a machine, ephemeral spawn workers, no forced order."""

    def __init__(self, machine: Machine, margin: float = 0.03,
                 sample_interval: float = 0.5, numba_cache_dir: str | None = None,
                 context_overhead: float = float("inf"), vram_fraction: float = 0.5,
                 time_multiple: float = 10.0, context_vram_reserve: int | None = None,
                 warm_setup=None, warm_between=None):
        self.machine = machine
        self.margin = margin
        self.sample_interval = sample_interval
        self.numba_cache_dir = numba_cache_dir
        # Warm-versus-fresh routing knobs. context_overhead defaults to infinity so the time
        # trigger stays off until it is measured on a real device; until then the decision is
        # vram-only. context_vram_reserve is the CUDA context's VRAM: held back once for the warm
        # worker and charged to every fresh GPU process. None means measure it on the device at
        # run start (when a GPU and warm hooks are present); an explicit int overrides the probe.
        # warm_setup/warm_between are the warm worker's session hooks; the GPU build supplies the
        # pair that holds a CUDA context and frees its pool, and None makes a plain serial worker.
        self.context_overhead = context_overhead
        self.vram_fraction = vram_fraction
        self.time_multiple = time_multiple
        self.context_vram_reserve = context_vram_reserve
        self.warm_setup = warm_setup
        self.warm_between = warm_between

    def _measure_overhead(self, ctx, platform, accelerator, session_setup):
        """The per-process overhead admission must charge: the per-worker RAM baseline (zero where
        workers share imports via fork, a measured cold-import cost where they spawn) and the
        device context's VRAM (charged per fresh GPU process). Both come from one probe that is
        what the heaviest worker is. An explicit context_vram_reserve overrides the device probe.
        Skips the probe entirely when there is nothing to measure (shared imports and no device
        VRAM to size). Fails loud if a probe that should report cannot."""
        worker_shared = platform.worker_memory_shared()
        explicit_context = self.context_vram_reserve is not None
        measure_context = (accelerator is not None and bool(self.machine.gpus)
                           and session_setup is not None and not explicit_context)
        if worker_shared and not measure_context:
            return 0, (self.context_vram_reserve if explicit_context else 0)

        probe_queue = ctx.Queue()
        proc = ctx.Process(target=_overhead_probe_main,
                           args=(probe_queue, platform, accelerator, session_setup))
        proc.start()
        try:
            measured = probe_queue.get(timeout=120)
        finally:
            proc.join()
        if measured is None:
            raise TetradromeError(
                "overhead probe failed; cannot size the per-worker RAM baseline and context "
                "VRAM this platform and device need.")
        probe_ram, probe_context = measured
        ram_baseline = 0 if worker_shared else probe_ram
        if explicit_context:
            context_vram = self.context_vram_reserve
        elif measure_context:
            context_vram = probe_context
        else:
            context_vram = 0
        return ram_baseline, context_vram

    def run(self, graph: JobGraph) -> RunReport:
        if self.numba_cache_dir:
            os.environ["NUMBA_CACHE_DIR"] = self.numba_cache_dir
        platform = for_host()
        logger.info("host platform: %s", platform.name)
        ctx = _start_context()
        result_queue = ctx.Queue()
        sample_queue: queue.Queue = queue.Queue()
        ledger = Ledger(self.machine)

        # The accelerator axis: which device vendor is present, if any. It supplies the warm
        # session hooks (a caller-provided pair still overrides) and the context-free VRAM read
        # the overhead probe needs.
        accelerator = detect_accelerator()
        warm_setup = self.warm_setup
        warm_between = self.warm_between
        if accelerator is not None:
            if warm_setup is None:
                warm_setup = accelerator.session_setup()
            if warm_between is None:
                warm_between = accelerator.session_between()

        # Per-process overhead the working-set footprint omits: a fresh worker's RAM baseline
        # (zero where workers fork and share imports, a measured cold-import cost where they
        # spawn) and the device context's VRAM (reserved once for the warm worker, charged to
        # every fresh GPU process). One probe measures both on the box.
        ram_baseline, context_vram = self._measure_overhead(ctx, platform, accelerator, warm_setup)
        worker_shared = platform.worker_memory_shared()
        if ram_baseline:
            logger.info("per-worker RAM baseline (spawn, charged per fresh worker): %d bytes",
                        ram_baseline)
        if context_vram:
            logger.info("CUDA context VRAM (charged per GPU process): %d bytes", context_vram)
        gpu_budget = self.machine.gpus[0].vram_bytes if self.machine.gpus else None

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
                                warm_setup, warm_between, platform)
            warm_workers[gpu_index] = worker
            if context_vram > 0 or ram_baseline > 0:
                # The warm worker is one standing process holding a CUDA context. Reserve its
                # context VRAM and (under spawn) its import baseline once, so admission never
                # hands out memory the worker itself is already using; its serial jobs then
                # charge only their own working set.
                ledger.add(Allocation(
                    job_key=("__warm_context__", gpu_index), placement=Placement.GPU,
                    cores=frozenset(), declared_ram=ram_baseline, node_index=node,
                    gpu_index=gpu_index, declared_vram=context_vram))
            return worker

        def launch(job, placed, gpu_execution) -> None:
            deps = {dep: results[dep] for dep in job.dependencies}
            placement = placed.path.placement
            # placed.path is the admission view: its footprint already includes the per-process
            # overhead for a fresh worker, and is the bare working set for a warm one.
            ledger.add(Allocation(
                job_key=job.key, placement=placement, cores=placed.cores,
                declared_ram=placed.path.ram_bytes, node_index=placed.node_index,
                gpu_index=placed.gpu_index, declared_vram=placed.path.vram_bytes,
            ))
            declared_ram[job.key] = placed.path.ram_bytes
            placement_of[job.key] = placement
            if placement is Placement.GPU and gpu_execution is Execution.WARM:
                # Small GPU job: serialize it through the held context. Not RAM-sampled -- it is
                # small by the gate, and only one warm job's footprint is ever live in the shared
                # pid at a time, so a live sample could not be attributed to it cleanly anyway.
                ensure_warm(placed.gpu_index).dispatch(job.key, job.run, job.inputs, deps)
                warm_running[job.key] = placed.gpu_index
            else:
                proc = ctx.Process(target=_worker_main, args=(
                    job.key, job.run, job.inputs, deps,
                    placed.cores, placed.gpu_index, result_queue, platform,
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
                logger.debug("job %r done: predicted %d, peak private %d",
                             key, declared_ram[key], peak_actual.get(key, 0))
            else:                       # warm: the worker persists, only the allocation clears
                warm_running.pop(key)
                ledger.remove(key)
                logger.debug("job %r done (warm)", key)
            if status == "ok":
                results[key] = payload
                completed.add(key)
                timings[key] = compute_time
                calibration.observe(float(graph.get(key).cost), placement_of[key], compute_time)
                logger.debug("job %r runtime %.4fs vs predicted cost %g",
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
                # Show the placer the fresh-worker footprint, not just the working set, so it
                # cannot admit more processes than the box can actually hold. The GPU path's
                # warm/fresh decision comes back so launch dispatches it the same way.
                predicted = calibration.predicted_time(float(job.cost), Placement.GPU)
                aug_paths, gpu_execution = _augment_for_admission(
                    job.paths, worker_shared=worker_shared, ram_baseline=ram_baseline,
                    context_vram=context_vram, gpu_budget=gpu_budget, predicted_gpu_time=predicted,
                    context_overhead=self.context_overhead, vram_fraction=self.vram_fraction,
                    time_multiple=self.time_multiple)
                aug_job = dataclasses.replace(job, paths=aug_paths)
                decision = plan_placement(self.machine, ledger, aug_job, self.margin)
                if decision.outcome is Outcome.ADMIT:
                    launch(job, decision.placed, gpu_execution)
                    did = True
            return did

        # Pre-flight feasibility: a job whose every declared path, with its own per-process
        # overhead folded in, overflows the bare machine can never run, no matter what frees up.
        # Decide it here, once, against the same augmented view the loop places against. Each
        # infeasible job poisons only its own lineage (component), so independent components still
        # run; every infeasible job is collected and surfaced in the report rather than raised, so
        # one bad job does not sink the batch.
        infeasible: list = []
        for job in graph.jobs():
            predicted = calibration.predicted_time(float(job.cost), Placement.GPU)
            aug_paths, _ = _augment_for_admission(
                job.paths, worker_shared=worker_shared, ram_baseline=ram_baseline,
                context_vram=context_vram, gpu_budget=gpu_budget, predicted_gpu_time=predicted,
                context_overhead=self.context_overhead, vram_fraction=self.vram_fraction,
                time_multiple=self.time_multiple)
            error = job_feasibility(self.machine, dataclasses.replace(job, paths=aug_paths),
                                    self.margin)
            if error is not None:
                infeasible.append(error)
        for error in infeasible:
            cancelled |= graph.component(error.job_key)
        if infeasible:
            logger.warning("pre-flight: %d job(s) infeasible, abandoning their lineage: %s",
                           len(infeasible), ", ".join(repr(e.job_key) for e in infeasible))

        stop = threading.Event()
        sampler = _Sampler(snapshot, sample_queue, self.sample_interval, stop, platform)
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
                if context_vram > 0 or ram_baseline > 0:
                    ledger.remove(("__warm_context__", gpu_index))
                worker.shutdown()
            result_queue.close()
            result_queue.join_thread()
        return RunReport(results=results, failures=failures,
                         cancelled=frozenset(cancelled), infeasible=tuple(infeasible),
                         timings=timings, calibration=calibration)
