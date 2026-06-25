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
import enum
import logging
import multiprocessing
import os
import pickle
import queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
from multiprocessing.shared_memory import SharedMemory
from typing import NamedTuple

from ..errors import TetradromeError
from .graph import JobGraph
from .hostplatform import for_host
from .accelerator import detect_accelerator
from .inventory import Machine
from .job import Placement, Shard
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


class SharedRef(NamedTuple):
    """A dependency handed to a worker by reference rather than value: its payload lives in a
    shared-memory segment the parent created and owns. The worker attaches the segment by name,
    reads ``size`` bytes, unpickles a private copy, and closes -- it never unlinks. Small and
    picklable, so only this reference rides into each worker, not a private copy of the payload."""
    name: str
    size: int


class DeviceRef(NamedTuple):
    """A dependency that lives in the warm worker's device memory under ``key``. Handed to a warm
    consumer in place of the value; the warm worker resolves it from its device registry and passes
    the resident buffer straight to ``run`` -- no host round trip. Only the warm worker can resolve
    it, since the buffer lives in its CUDA context."""
    key: object


class DeviceHandle(NamedTuple):
    """Posted by the warm worker as a device-resident producer's result: the buffer stayed in the
    worker's registry under ``key`` and only this handle (with its ``nbytes`` for the VRAM charge)
    travels to the parent."""
    key: object
    nbytes: int


def _attach_segment(name):
    """Attach a shared segment created elsewhere, for reading only. The creating parent owns the
    segment and is the sole unlinker. Within one multiprocessing program the resource_tracker is
    shared, so the parent's unlink already unregisters the name -- an attaching worker must not
    unregister (that would remove the entry first and make the parent's unlink fail) and must not
    unlink. Python 3.13's ``track=False`` keeps the attach out of the tracker entirely; on 3.12 a
    plain attach only adds a duplicate the parent's unlink clears, and the worker never unlinks."""
    try:
        return SharedMemory(name=name, track=False)          # 3.13+: attach untracked
    except TypeError:
        return SharedMemory(name=name)                       # <=3.12: parent's unlink cleans up


def _resolve_deps(deps):
    """Replace any SharedRef in a worker's deps with a private copy read from its segment. Attach,
    read the exact byte range, unpickle, and close immediately -- the unpickled object is the
    worker's own; the segment stays owned by the parent."""
    resolved = {}
    for dep_key, value in deps.items():
        if isinstance(value, SharedRef):
            shm = _attach_segment(value.name)
            try:
                resolved[dep_key] = pickle.loads(bytes(shm.buf[:value.size]))
            finally:
                shm.close()
        else:
            resolved[dep_key] = value
    return resolved


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
        deps = _resolve_deps(deps)            # pull any shared-segment deps into private copies
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
    device_resident: dict = {}             # key -> buffer kept on the device across jobs
    while True:
        item = job_queue.get()
        if item is None:
            return
        if item[0] == "free":              # parent says a device-resident result's last reader is done
            device_resident.pop(item[1], None)
            continue
        _, key, run, inputs, deps, keep = item
        try:
            deps = _resolve_deps(deps)        # pull any shared-segment deps into private copies
            deps = {dep_key: (device_resident[value.key] if isinstance(value, DeviceRef) else value)
                    for dep_key, value in deps.items()}
            t0 = time.perf_counter()
            result = run(inputs, deps)
            elapsed = time.perf_counter() - t0
            if keep:                          # device-resident output: keep the buffer, post a handle
                device_resident[key] = result
                result_queue.put((key, "ok", DeviceHandle(key, result.nbytes), elapsed))
            else:
                result_queue.put((key, "ok", result, elapsed))
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

    def dispatch(self, key, run, inputs, deps, keep_on_device=False) -> None:
        self._job_queue.put(("job", key, run, inputs, deps, keep_on_device))

    def free_device(self, key) -> None:
        # Tell the worker a device-resident result's last reader has finished, so it can drop the
        # buffer. Queued after that reader's job, and the queue is serial, so the read completes
        # before the free runs.
        self._job_queue.put(("free", key))

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


class Residence(enum.Enum):
    """Where a resident blob physically lives. HOST_PRIVATE and DISK are the everyday two;
    HOST_SHARED is one copy in shared host RAM that many consumers map instead of each getting a
    private pickled copy (the base form is a single segment, and per-NUMA-node replicas layer on
    for pinned consumers); DEVICE (resident in GPU VRAM) is reserved so adding it fills a slot
    rather than reshaping the table."""
    HOST_PRIVATE = "host_private"   # a private copy in the parent's RAM (the default)
    DISK = "disk"                   # spilled to a file under memory pressure
    HOST_SHARED = "host_shared"     # one shared-RAM segment many consumers map (per locality domain)
    DEVICE = "device"               # resident in a warm worker's GPU VRAM, read on-device


class Kind(enum.Enum):
    """What a resident blob is, which fixes how its lifetime ends."""
    OUTPUT = "output"               # a held result, freed when its last dependent is dispatched
    INPUT = "input"                 # a static input, freed when its own job is dispatched


@dataclasses.dataclass
class Resident:
    """One resident blob -- a held result or a detached static input -- tracked uniformly. The
    residence says where it is and therefore how to free it (drop the RAM charge if HOST_PRIVATE,
    delete the file if DISK); size is the charged bytes either way; payload holds the value while
    HOST_PRIVATE and location the file path while DISK; consumers counts dependents still to be
    dispatched for an OUTPUT and is unused for an INPUT (consumed once at its own dispatch)."""
    kind: Kind
    residence: Residence
    size: int
    payload: object = None
    location: str | None = None
    consumers: int = 0
    shareable: bool = False         # fan-out and size cleared the threshold: share rather than copy
    handle: object = None           # HOST_SHARED: {domain: SharedMemory}, one segment per locality
                                    # domain (None = base, or a NUMA node index); reserved for DEVICE
    gpu_index: int | None = None    # DEVICE: which GPU's warm worker holds the resident buffer


@dataclasses.dataclass
class RunReport:
    """The outcome of a run: results for completed jobs, one entry per failed component, and one
    per job found infeasible up front.

    ``timings`` maps each completed job's key to the seconds its ``run`` took, the measured work
    behind the cost model: comparing it to the job's predicted ``cost`` is what calibrates the
    per-op constant and, in turn, the warm-versus-fresh routing. ``infeasible`` holds one
    InfeasibleJobError per job the bare machine could serve on no declared path; their lineages are
    abandoned but the rest of the batch runs. ``spill_count``/``spilled_bytes`` record the degraded
    path: how many held results were written to disk under memory pressure, and their total size. A
    nonzero count means the run completed only by spilling. ``shared_count``/``shared_bytes`` record
    how many held results were materialized into a shared-memory segment (high fan-out above the
    size floor) and their total segment size -- the transmit-once path, not a degraded one.
    """
    results: dict
    failures: list          # (component frozenset, failed_key, error_text)
    cancelled: frozenset
    infeasible: tuple = ()  # InfeasibleJobError per job infeasible on the bare machine
    spilled_bytes: int = 0  # total held-result bytes written to disk under pressure (0 = none)
    spill_count: int = 0    # how many held results were spilled (the degraded path fired if > 0)
    shared_count: int = 0   # how many results were materialized into a shared-memory segment
    shared_bytes: int = 0   # total size of those segments
    device_count: int = 0   # how many results were kept resident in GPU VRAM
    device_bytes: int = 0   # total VRAM bytes of those device-resident results
    timings: dict = dataclasses.field(default_factory=dict)
    calibration: "Calibration | None" = None     # final per-placement rates, for inspection


class Scheduler:
    """Runs a JobGraph on a machine, ephemeral spawn workers, no forced order."""

    def __init__(self, machine: Machine, margin: float = 0.03,
                 sample_interval: float = 0.5, numba_cache_dir: str | None = None,
                 context_overhead: float = float("inf"), vram_fraction: float = 0.5,
                 time_multiple: float = 10.0, context_vram_reserve: int | None = None,
                 warm_setup=None, warm_between=None,
                 spill_dir: str | None = None, spill_floor_bytes: int = 16 * 1024 * 1024,
                 spill_budget_bytes: int = 4 * 1024 * 1024 * 1024,
                 shared_min_consumers: int = 2, shared_floor_bytes: int = 16 * 1024 * 1024):
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
        # Spill: under memory pressure, the largest held results above the floor are written to
        # disk to free RAM rather than stalling. The floor keeps spill to heavy outputs so small,
        # fast results never thrash through disk; the budget caps total on-disk spill. spill_dir
        # None means a per-run directory under the system temp.
        self.spill_dir = spill_dir
        self.spill_floor_bytes = spill_floor_bytes
        self.spill_budget_bytes = spill_budget_bytes
        # Shared residence: a held result with at least this many consumers and at least this many
        # bytes is materialized once into a shared-memory segment the consumers map, instead of a
        # private pickled copy per consumer (serialize once, transmit once). The defaults are
        # conservative -- multi-consumer and large -- so small or low-fan-out results stay private;
        # both are policy knobs to sweep, not heuristics.
        self.shared_min_consumers = shared_min_consumers
        self.shared_floor_bytes = shared_floor_bytes

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
        output_over_warned: set = set()  # keys already warned for an oversized result
        # One table for every resident blob -- held results and detached static inputs alike --
        # keyed by (kind, key). Each entry records its residence, so freeing, spilling, and
        # reporting are one uniform mechanism rather than four parallel dicts.
        resident: dict = {}
        spill_state = {"dir": None, "count": 0, "total_bytes": 0, "disk_used": 0}
        shared_state = {"count": 0, "bytes": 0}     # results materialized into shared segments
        device_state = {"count": 0, "bytes": 0}     # results kept resident in GPU VRAM
        nodes_by_index = {node.index: node for node in self.machine.nodes}
        replica_stall_warned: set = set()           # result keys already warned for stalling dispatch
        placement_of: dict = {}         # key -> the placement it ran on, for calibration
        calibration = Calibration()
        timings: dict = {}              # key -> measured run seconds, for cost calibration
        completed: set = set()
        cancelled: set = set()
        failures: list = []

        # Take ownership of the jobs' inputs so they are no longer pinned by the (frozen) jobs in
        # the graph. Each is charged against global RAM and freed when its job is dispatched (an
        # input is consumed exactly once), so the resident input set shrinks as the run proceeds
        # instead of every input staying pinned for the whole run. Heavy ones spill like outputs.
        inputs_store = graph.detach_inputs()
        for input_key, input_data in inputs_store.items():
            size = len(pickle.dumps(input_data))
            resident[(Kind.INPUT, input_key)] = Resident(
                kind=Kind.INPUT, residence=Residence.HOST_PRIVATE, size=size, payload=input_data)
            ledger.add(Allocation(
                job_key=("__input__", input_key), placement=Placement.CPU_UNPINNED,
                cores=frozenset(), declared_ram=size, node_index=None))
        del inputs_store                # the table is the sole owner now

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

        def release_resident(table_key) -> None:
            # Drop one resident blob: a HOST_PRIVATE one releases its global-RAM charge, a DISK one
            # deletes its spill file and reclaims the disk budget, a HOST_SHARED one (last consumer
            # dispatched) closes and unlinks its segment -- the parent is the sole unlinker. No-op if
            # already gone.
            entry = resident.pop(table_key, None)
            if entry is None:
                return
            kind, key = table_key
            charge = "__input__" if kind is Kind.INPUT else "__output__"
            if entry.residence is Residence.DISK:
                os.remove(entry.location)
                spill_state["disk_used"] -= entry.size
            elif entry.residence is Residence.HOST_SHARED:
                for domain, segment in entry.handle.items():
                    ledger.discard(("__output__", (key, domain)))
                    segment.close()
                    segment.unlink()
            elif entry.residence is Residence.DEVICE:
                ledger.discard(("__output__", key))
                worker = warm_workers.get(entry.gpu_index)
                if worker is not None:
                    worker.free_device(key)         # drop the buffer from the worker's registry
            else:
                ledger.discard((charge, key))

        def _warn_if_stalling(key, domain) -> None:
            # Materializing a replica is a synchronous write in the main loop, so it delays
            # dispatch of everything else. Warn (once per result) when that write coincides with
            # work that could be running: a ready job not yet started and a node with both free
            # cores and free RAM. It can fire when the specific waiting job would not fit the
            # specific free node; that is the deliberate loud-erring side, to surface on labradorite.
            if key in replica_stall_warned:
                return
            has_waiting = any(
                not is_running(job.key) and job.key not in cancelled and job.key != key
                for job in graph.ready(completed))
            if not has_waiting:
                return
            has_free = any(ledger.free_cores(node.index) and ledger.free_ram_node(node.index) > 0
                           for node in self.machine.nodes)
            if not has_free:
                return
            replica_stall_warned.add(key)
            where = "the base segment" if domain is None else f"node {domain}"
            logger.warning("replicating shared result %r to %s stalled dispatch while ready jobs "
                           "and free NUMA capacity were available", key, where)

        def _make_replica(entry, key, domain, source):
            # Create one locality domain's segment for a shared result and charge it. A node domain
            # writes inside local_to so its pages first-touch on that node and is charged against
            # that node's RAM; the base domain (None) writes unbound and is charged globally. source
            # is the bytes (first replica) or an existing replica's buffer (a node-local copy of one
            # already made). Each replica is real RAM, charged at full size -- locality is not free.
            _warn_if_stalling(key, domain)
            segment = SharedMemory(create=True, size=entry.size)
            if domain is None:
                segment.buf[:entry.size] = source
                node_index = None
            else:
                with platform.local_to(nodes_by_index[domain].cores):
                    segment.buf[:entry.size] = source
                node_index = domain
            ledger.add(Allocation(
                job_key=("__output__", (key, domain)), placement=Placement.CPU_UNPINNED,
                cores=frozenset(), declared_ram=entry.size, node_index=node_index))
            entry.handle[domain] = segment
            shared_state["count"] += 1
            shared_state["bytes"] += entry.size
            assert len(entry.handle) <= len(self.machine.nodes) + 1, "more replicas than locality domains"
            return segment

        def _load_dep(dep, domain):
            # A consumer's view of a producer's result. A shareable result is materialized into a
            # shared-memory segment on the first consumer's dispatch (serialize once, drop the
            # private copy); thereafter each consumer is handed a small SharedRef to a segment in
            # its locality domain -- its NUMA node for a pinned consumer (a local replica, copied
            # from an existing one if new), or any existing replica for an unpinned one, which has
            # no locality to honor. A plain result is copied; a spilled one is read from disk.
            entry = resident[(Kind.OUTPUT, dep)]
            if entry.residence is Residence.DEVICE:
                return DeviceRef(dep)               # warm consumer resolves it from the worker registry
            if entry.residence is Residence.HOST_SHARED:
                if domain in entry.handle:
                    segment = entry.handle[domain]
                elif domain is None:
                    segment = next(iter(entry.handle.values()))     # unpinned: reuse any replica
                else:
                    segment = _make_replica(entry, dep, domain,
                                            next(iter(entry.handle.values())).buf[:entry.size])
                return SharedRef(segment.name, entry.size)
            if entry.residence is Residence.DISK:
                with open(entry.location, "rb") as f:
                    return pickle.load(f)
            if entry.shareable:
                data = pickle.dumps(entry.payload)
                entry.residence = Residence.HOST_SHARED
                entry.payload = None
                entry.size = len(data)                  # exact segment bytes the readers slice to
                entry.handle = {}
                ledger.discard(("__output__", dep))     # recharge per replica, not as one whole copy
                segment = _make_replica(entry, dep, domain, data)
                if shared_state["count"] == 1:
                    logger.info("sharing result %r across %d consumers via %d-byte segment(s)",
                                dep, entry.consumers, len(data))
                return SharedRef(segment.name, entry.size)
            return entry.payload

        def _load_input(key):
            # The job's input lives in the resident table (it was detached from the job), on disk if
            # it was spilled. Read back transiently to feed the worker.
            entry = resident[(Kind.INPUT, key)]
            if entry.residence is Residence.DISK:
                with open(entry.location, "rb") as f:
                    return pickle.load(f)
            return entry.payload

        def launch(job, placed, gpu_execution) -> None:
            # A pinned consumer maps a replica local to its node; everyone else (unpinned, GPU) maps
            # the base segment, which has no node to honor.
            placement = placed.path.placement
            warm = placement is Placement.GPU and gpu_execution is Execution.WARM
            # Device residence is a performance optimization, not a correctness requirement: a buffer
            # lives in the warm worker's context, so it is only useful when producer and consumers
            # all run there. When routing defeats that, warn about the degradation and keep computing
            # rather than aborting -- the result is still produced, just on the host or via a
            # serialized warm run. The producer keeps its result resident only when it runs warm and
            # every consumer is GPU-only (a consumer that can fall to the CPU could not read a device
            # buffer, so the result goes to host instead).
            consumers_all_gpu = all(path.placement is Placement.GPU
                                    for consumer in graph.dependents(job.key)
                                    for path in graph.get(consumer).paths)
            keep_on_device = job.device_resident and warm and consumers_all_gpu
            if job.device_resident and not keep_on_device:
                why = "it routed fresh rather than warm" if not warm else "a consumer is not GPU-only"
                logger.warning("device-resident job %r: %s, so its result is host-resident this run "
                               "(device residency skipped)", job.key, why)
            if not warm and placement is Placement.GPU and any(
                    resident[(Kind.OUTPUT, dep)].residence is Residence.DEVICE
                    for dep in job.dependencies):
                logger.warning("job %r reads a device-resident result but routed fresh; running it "
                               "warm to read the buffer in-context, which serializes it on the GPU",
                               job.key)
                warm = True
            domain = placed.node_index if placement is Placement.CPU_PINNED else None
            deps = {dep: _load_dep(dep, domain) for dep in job.dependencies}
            job_inputs = _load_input(job.key)
            # placed.path is the admission view: its footprint already includes the per-process
            # overhead for a fresh worker, and is the bare working set for a warm one.
            ledger.add(Allocation(
                job_key=job.key, placement=placement, cores=placed.cores,
                declared_ram=placed.path.ram_bytes, node_index=placed.node_index,
                gpu_index=placed.gpu_index, declared_vram=placed.path.vram_bytes,
            ))
            declared_ram[job.key] = placed.path.ram_bytes
            placement_of[job.key] = placement
            if warm:
                # Small GPU job: serialize it through the held context. Not RAM-sampled -- it is
                # small by the gate, and only one warm job's footprint is ever live in the shared
                # pid at a time, so a live sample could not be attributed to it cleanly anyway. A
                # device-resident job keeps its result in the worker's VRAM rather than returning it.
                ensure_warm(placed.gpu_index).dispatch(
                    job.key, job.run, job_inputs, deps, keep_on_device=keep_on_device)
                warm_running[job.key] = placed.gpu_index
            else:
                proc = ctx.Process(target=_worker_main, args=(
                    job.key, job.run, job_inputs, deps,
                    placed.cores, placed.gpu_index, result_queue, platform,
                ))
                proc.start()
                running[job.key] = proc
                with pids_lock:
                    running_pids[job.key] = proc.pid
            if placed.note:
                logger.info("job %r degraded: %s", job.key, placed.note)
            # The input has been copied into the worker, so its parent-side copy is no longer
            # needed -- this job is its only consumer. Release it.
            release_resident((Kind.INPUT, job.key))
            # The deps have been copied into the worker (started or dispatched), so each producer's
            # held result (or shard) is no longer needed by this consumer -- except a by-reference
            # dep (a shared segment, or a device-resident buffer), which the worker reads during its
            # run, not at dispatch, so it is freed at the consumer's completion instead (see
            # reap_one). Drop a copied dep once its last consumer has been dispatched.
            for dep in job.dependencies:
                entry = resident[(Kind.OUTPUT, dep)]
                if entry.residence in (Residence.HOST_SHARED, Residence.DEVICE):
                    continue
                entry.consumers -= 1
                if entry.consumers == 0:
                    release_resident((Kind.OUTPUT, dep))

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
                release_resident((Kind.OUTPUT, member))   # whole result, if it produced one
                release_resident((Kind.INPUT, member))    # its undispatched input, if any
                member_shards = graph.get(member).shards
                if member_shards is not None:             # a partitioned producer's shards
                    for shard_key in member_shards:
                        release_resident((Kind.OUTPUT, Shard(member, shard_key)))
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
            gpu_completed = None        # the GPU a warm job ran on, kept for a device-resident result
            if key in running:          # fresh: its own process to reap
                proc = running.pop(key)
                with pids_lock:
                    running_pids.pop(key, None)
                ledger.remove(key)
                proc.join()
                logger.debug("job %r done: predicted %d, peak private %d",
                             key, declared_ram[key], peak_actual.get(key, 0))
            else:                       # warm: the worker persists, only the allocation clears
                gpu_completed = warm_running.pop(key)
                ledger.remove(key)
                logger.debug("job %r done (warm)", key)
            if status == "ok":
                completed.add(key)
                timings[key] = compute_time
                calibration.observe(float(graph.get(key).cost), placement_of[key], compute_time)
                # The result persists in the parent until its consumers drain it, so charge it
                # against global RAM like a working-set footprint. A whole result is one held entry;
                # a partitioned one fans into a held entry per shard, each charged by its own
                # measured size and ref-counted by the consumers depending on that shard, so a wide
                # consumer never holds the producer's whole output. The over-budget warning compares
                # the declared output_bytes (a total, for a partitioned job) against the measured
                # total, so a too-low estimate tunes rather than under-charges silently.
                declared_out = graph.get(key).output_bytes
                job_shards = graph.get(key).shards
                if isinstance(payload, DeviceHandle):
                    # A device-resident result: the buffer stayed in the warm worker's VRAM and only
                    # the handle came back. Track it as DEVICE, charged against that GPU's VRAM and
                    # ref-counted by its consumers; the warm worker frees the buffer when the parent
                    # signals the last reader is done (release_resident).
                    fanout = len(graph.dependents(key))
                    ledger.add(Allocation(
                        job_key=("__output__", key), placement=Placement.GPU, cores=frozenset(),
                        declared_ram=0, gpu_index=gpu_completed, declared_vram=payload.nbytes))
                    resident[(Kind.OUTPUT, key)] = Resident(
                        kind=Kind.OUTPUT, residence=Residence.DEVICE, size=payload.nbytes,
                        consumers=fanout, gpu_index=gpu_completed)
                    device_state["count"] += 1
                    device_state["bytes"] += payload.nbytes
                    logger.info("result %r resident in VRAM on gpu %s (%d bytes), %d consumer(s)",
                                key, gpu_completed, payload.nbytes, fanout)
                elif job_shards is None:
                    measured_out = len(pickle.dumps(payload))
                    size = max(declared_out, measured_out)
                    fanout = len(graph.dependents(key))
                    ledger.add(Allocation(
                        job_key=("__output__", key), placement=Placement.CPU_UNPINNED,
                        cores=frozenset(), declared_ram=size, node_index=None))
                    resident[(Kind.OUTPUT, key)] = Resident(
                        kind=Kind.OUTPUT, residence=Residence.HOST_PRIVATE, size=size,
                        payload=payload, consumers=fanout,
                        shareable=(fanout >= self.shared_min_consumers
                                   and size >= self.shared_floor_bytes))
                else:
                    if not isinstance(payload, dict) or set(payload) != set(job_shards):
                        raise TetradromeError(
                            f"partitioned job {key!r} must return exactly its declared shards "
                            f"{set(job_shards)}; got "
                            f"{set(payload) if isinstance(payload, dict) else type(payload).__name__}")
                    consumers = {c: graph.get(c).dependencies for c in graph.dependents(key)}
                    measured_out = 0
                    for shard_key, shard_payload in payload.items():
                        shard_ref = Shard(key, shard_key)
                        shard_size = len(pickle.dumps(shard_payload))
                        shard_fanout = sum(1 for deps in consumers.values() if shard_ref in deps)
                        measured_out += shard_size
                        ledger.add(Allocation(
                            job_key=("__output__", shard_ref), placement=Placement.CPU_UNPINNED,
                            cores=frozenset(), declared_ram=shard_size, node_index=None))
                        resident[(Kind.OUTPUT, shard_ref)] = Resident(
                            kind=Kind.OUTPUT, residence=Residence.HOST_PRIVATE, size=shard_size,
                            payload=shard_payload, consumers=shard_fanout,
                            shareable=(shard_fanout >= self.shared_min_consumers
                                       and shard_size >= self.shared_floor_bytes))
                if declared_out and measured_out > declared_out and key not in output_over_warned:
                    output_over_warned.add(key)
                    logger.warning("job %r output exceeded declared budget: declared %d, actual %d",
                                   key, declared_out, measured_out)
                # This job has finished, so it has finished reading any by-reference deps (read at
                # the start of its run): shared segments and device-resident buffers. Decrement
                # those; the segment or buffer is released once its last reader completes. Copied
                # deps were already released at dispatch.
                for dep in graph.get(key).dependencies:
                    entry = resident.get((Kind.OUTPUT, dep))
                    if entry is not None and entry.residence in (Residence.HOST_SHARED, Residence.DEVICE):
                        entry.consumers -= 1
                        if entry.consumers == 0:
                            release_resident((Kind.OUTPUT, dep))
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

        def _spill_dir() -> str:
            if spill_state["dir"] is None:
                spill_state["dir"] = tempfile.mkdtemp(prefix="tetradrome-spill-",
                                                      dir=self.spill_dir)
            return spill_state["dir"]

        def spill_to_make_room() -> bool:
            # Free RAM by writing the largest HOST_PRIVATE blob above the floor that still fits the
            # disk budget out to disk, releasing its global-RAM charge. Held results and the resident
            # inputs of not-yet-dispatched jobs are eligible alike; heavy blobs only (the floor) and
            # largest first, so each spill reclaims the most RAM for one write and small, fast data
            # never thrashes through disk. Returns whether one was spilled; False means nothing is
            # left to spill, the true all-tiers-full dead-end.
            candidates = sorted(
                ((tablekey, entry) for tablekey, entry in resident.items()
                 if entry.residence is Residence.HOST_PRIVATE and entry.size >= self.spill_floor_bytes),
                key=lambda item: item[1].size, reverse=True)
            for tablekey, entry in candidates:
                if spill_state["disk_used"] + entry.size > self.spill_budget_bytes:
                    continue                     # this heavy blob will not fit the disk budget
                kind, key = tablekey
                path = os.path.join(_spill_dir(), f"{spill_state['count']:06d}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(entry.payload, f)
                charge = "__input__" if kind is Kind.INPUT else "__output__"
                ledger.remove((charge, key))
                entry.residence = Residence.DISK
                entry.location = path
                entry.payload = None
                spill_state["disk_used"] += entry.size
                spill_state["total_bytes"] += entry.size
                spill_state["count"] += 1
                if spill_state["count"] == 1:
                    logger.warning("memory pressure: spilling %s to disk (degraded path); first "
                                   "spill %r, %d bytes", kind.value, key, entry.size)
                return True
            return False

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
                elif spill_to_make_room():
                    continue            # freed RAM by spilling a heavy held result; retry admission
                else:
                    remaining = [job.key for job in graph.jobs()
                                 if job.key not in completed and job.key not in cancelled]
                    raise TetradromeError(
                        "scheduler stalled with nothing running and nothing left to spill "
                        f"(RAM full and disk budget exhausted); remaining: {remaining}")
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
            # Unlink any shared segment still resident: on a clean run every shared result is freed
            # as its last consumer dispatches, so this catches only the abort path, where leaving
            # POSIX segments around would leak them.
            for entry in resident.values():
                if entry.residence is Residence.HOST_SHARED and entry.handle:
                    for segment in entry.handle.values():
                        segment.close()
                        segment.unlink()
            if spill_state["dir"] is not None:
                shutil.rmtree(spill_state["dir"], ignore_errors=True)
        # Surviving held results: terminals never drained, plus any producer whose poisoned
        # consumers were never dispatched. As before, a spilled result is not restored here -- the
        # report is the RAM-resident ones (the spilled-terminal edge is a separate, deliberate fix).
        final_results = {key: entry.payload
                         for (kind, key), entry in resident.items()
                         if kind is Kind.OUTPUT and entry.residence is Residence.HOST_PRIVATE}
        return RunReport(results=final_results, failures=failures,
                         cancelled=frozenset(cancelled), infeasible=tuple(infeasible),
                         spilled_bytes=spill_state["total_bytes"],
                         spill_count=spill_state["count"],
                         shared_count=shared_state["count"],
                         shared_bytes=shared_state["bytes"],
                         device_count=device_state["count"],
                         device_bytes=device_state["bytes"],
                         timings=timings, calibration=calibration)
