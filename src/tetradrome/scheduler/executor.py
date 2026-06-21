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
import threading
import traceback

from ..errors import TetradromeError
from .graph import JobGraph
from .inventory import Machine
from .ledger import Allocation, Ledger
from .placement import Outcome, plan_placement

logger = logging.getLogger(__name__)

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_GRACE_SECONDS = 2.0


def _worker_main(key, run, inputs, deps, cores, gpu_index, result_queue):
    """Top-level worker entry, picklable for spawn: pin, run, report exactly one message."""
    try:
        os.sched_setaffinity(0, set(cores))
        if gpu_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        result = run(inputs, deps)
    except Exception:
        result_queue.put((key, "error", traceback.format_exc()))
        return
    result_queue.put((key, "ok", result))


def _read_rss(pid: int) -> int | None:
    """Resident bytes for a pid, or None if the process is gone or unreadable."""
    try:
        with open(f"/proc/{pid}/statm") as handle:
            resident_pages = int(handle.read().split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None
    return resident_pages * _PAGE_SIZE


class _Sampler(threading.Thread):
    """A pure reader: snapshot the running pids, read each RSS, post to the loop's queue."""

    def __init__(self, snapshot, sink: queue.Queue, interval: float, stop: threading.Event):
        super().__init__(daemon=True)
        self._snapshot = snapshot
        self._sink = sink
        self._interval = interval
        self._stopping = stop

    def run(self) -> None:
        while not self._stopping.is_set():
            for key, pid in self._snapshot().items():
                rss = _read_rss(pid)
                if rss is not None:
                    self._sink.put((key, rss))
            self._stopping.wait(self._interval)


@dataclasses.dataclass
class RunReport:
    """The outcome of a run: results for completed jobs, and one entry per failed component."""
    results: dict
    failures: list          # (component frozenset, failed_key, error_text)
    cancelled: frozenset


class Scheduler:
    """Runs a JobGraph on a machine, ephemeral spawn workers, no forced order."""

    def __init__(self, machine: Machine, margin: float = 0.03,
                 sample_interval: float = 0.5, numba_cache_dir: str | None = None):
        self.machine = machine
        self.margin = margin
        self.sample_interval = sample_interval
        self.numba_cache_dir = numba_cache_dir

    def run(self, graph: JobGraph) -> RunReport:
        if self.numba_cache_dir:
            os.environ["NUMBA_CACHE_DIR"] = self.numba_cache_dir
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        sample_queue: queue.Queue = queue.Queue()
        ledger = Ledger(self.machine)

        running: dict = {}              # key -> Process
        running_pids: dict = {}         # key -> pid, shared with the sampler under the lock
        pids_lock = threading.Lock()
        declared_ram: dict = {}         # key -> declared peak, for warnings and the summary
        peak_actual: dict = {}          # key -> max sampled rss
        over_warned: set = set()        # keys already warned for crossing their declaration
        results: dict = {}
        completed: set = set()
        cancelled: set = set()
        failures: list = []

        def snapshot() -> dict:
            with pids_lock:
                return dict(running_pids)

        def launch(job, placed) -> None:
            deps = {dep: results[dep] for dep in job.dependencies}
            ledger.add(Allocation(
                job_key=job.key, placement=placed.path.placement, cores=placed.cores,
                declared_ram=placed.path.ram_bytes, node_index=placed.node_index,
                gpu_index=placed.gpu_index, declared_vram=placed.path.vram_bytes,
            ))
            proc = ctx.Process(target=_worker_main, args=(
                job.key, job.run, job.inputs, deps,
                placed.cores, placed.gpu_index, result_queue,
            ))
            proc.start()
            running[job.key] = proc
            declared_ram[job.key] = placed.path.ram_bytes
            with pids_lock:
                running_pids[job.key] = proc.pid
            if placed.note:
                logger.info("job %r degraded: %s", job.key, placed.note)

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
                    key, rss = sample_queue.get_nowait()
                except queue.Empty:
                    break
                got = True
                if key not in running:
                    continue
                peak_actual[key] = max(peak_actual.get(key, 0), rss)
                ledger.set_actual(key, rss)
                if rss > declared_ram[key] and key not in over_warned:
                    over_warned.add(key)
                    logger.warning("job %r exceeded declared RAM: declared %d, actual %d",
                                   key, declared_ram[key], rss)
            return got

        def reap_one(item) -> None:
            key, status, payload = item
            if key not in running:
                return                  # stale message from a job already reaped or killed
            proc = running.pop(key)
            with pids_lock:
                running_pids.pop(key, None)
            ledger.remove(key)
            proc.join()
            logger.info("job %r done: declared %d, peak actual %d",
                        key, declared_ram[key], peak_actual.get(key, 0))
            if status == "ok":
                results[key] = payload
                completed.add(key)
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
                if job.key in running or job.key in cancelled:
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
                if running:
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
            result_queue.close()
            result_queue.join_thread()
        return RunReport(results=results, failures=failures, cancelled=frozenset(cancelled))
