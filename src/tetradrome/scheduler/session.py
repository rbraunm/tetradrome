# SPDX-License-Identifier: Apache-2.0

"""The engines' way into the scheduler: run a graph on this machine and return what it computed.

Every engine calculation enters through ``run_to_result`` (one calculation) or ``run_many``
(several independent ones sharing the machine), so one calibration serves the whole process. It
holds the benchmark's figures for this machine and software; the first calculation makes sure
they exist, measuring them if not (``tetradrome.calibrate``). The machine is detected once per
process for the same reason a small calculation can afford to come through here at all: probing
the topology costs more than many of them do.
"""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import dataclasses
import logging

from ..errors import TetradromeError
from . import calibration_store
from .executor import Scheduler, WorkerPool
from .graph import JobGraph
from .inventory import Machine, detect_machine
from .job import scoped
from .routing import Calibration

logger = logging.getLogger(__name__)

_calibration = Calibration()
_machine: Machine | None = None
# Read once with the machine: reading the processor model costs milliseconds in a container.
_machine_identity: dict | None = None
_worker: dict = {}
# Kept between runs, so a calculation after the first finds workers waiting.
_pool = WorkerPool()
atexit.register(_pool.close)


@dataclasses.dataclass(frozen=True)
class JobRecord:
    """How one job ran: its key, its kind (what it computes and, for a reduction, the reducer),
    where the scheduler placed it (inline, a pinned or unpinned CPU worker, or a GPU), the CPUs
    it held, its NUMA node and GPU, and its run time in seconds."""

    job: str
    kind: str
    placement: str
    cpus: tuple
    node: int | None
    gpu: int | None
    seconds: float


@dataclasses.dataclass(frozen=True)
class ComputationRecord:
    """Every job one calculation ran, in the order they finished, on which machine (the
    benchmark's identity of it: processor, NUMA nodes and physical cores, GPUs), and the digest
    of the package source that ran them."""

    machine: dict
    source: str
    jobs: tuple


class Recorder:
    """Collects the jobs of every run made while it is recording."""

    def __init__(self):
        self.jobs: list = []

    def record(self) -> ComputationRecord:
        if _machine_identity is None:
            raise TetradromeError("nothing was scheduled while recording")
        return ComputationRecord(_machine_identity,
                                 calibration_store.software_identity()["source"],
                                 tuple(self.jobs))


_recorder: contextvars.ContextVar = contextvars.ContextVar("tetradrome_recorder", default=None)


@contextlib.contextmanager
def recording():
    """Within the block, every scheduler run's jobs are kept in the yielded ``Recorder``."""
    recorder = Recorder()
    token = _recorder.set(recorder)
    try:
        yield recorder
    finally:
        _recorder.reset(token)


def capped_machine(mem_cap_bytes: int | None, vram_cap_bytes: int | None) -> Machine:
    """The detected machine, optionally lowered to a tighter system-RAM and/or per-GPU VRAM
    ceiling. A cap only ever lowers a limit; it never promises more than the hardware has."""
    global _machine, _machine_identity
    if _machine is None:
        from ..calibrate import ensure          # the benchmark imports the engines, which import this
        machine = detect_machine()
        figures = ensure(_calibration, machine)
        _worker.update(worker_ram_baseline=figures["worker"]["ram_baseline"],
                       context_vram_reserve=figures["worker"]["context_vram"],
                       worker_start_seconds=figures["worker"]["start_seconds"])
        _machine_identity = calibration_store.machine_identity(machine)
        _machine = machine
    if mem_cap_bytes is None and vram_cap_bytes is None:
        return _machine
    capped_ram = (_machine.mem_cap_bytes if mem_cap_bytes is None
                  else min(_machine.mem_cap_bytes, mem_cap_bytes))
    gpus = _machine.gpus
    if vram_cap_bytes is not None:
        gpus = tuple(dataclasses.replace(gpu, vram_bytes=min(gpu.vram_bytes, vram_cap_bytes))
                     for gpu in _machine.gpus)
    return dataclasses.replace(_machine, mem_cap_bytes=capped_ram, gpus=gpus)


def run_to_result(graph, key, *, mem_cap_bytes: int | None = None,
                  vram_cap_bytes: int | None = None):
    """Run ``graph`` on the (optionally capped) machine and return the result stored under
    ``key``, failing loud: an infeasible job, a failed component, or a missing result all raise
    rather than hand back a partial or silently-empty answer."""
    report = _run(graph, mem_cap_bytes, vram_cap_bytes)
    if report.infeasible:
        raise report.infeasible[0]
    if report.failures:
        _, failed, text = report.failures[0]
        raise TetradromeError(f"computation failed at {failed!r}: {text}")
    if key not in report.results:
        raise TetradromeError(f"scheduler returned no result for {key!r}")
    return report.results[key]


def _run(graph, mem_cap_bytes, vram_cap_bytes, on_complete=None):
    machine = capped_machine(mem_cap_bytes, vram_cap_bytes)
    scheduler = Scheduler(machine, calibration=_calibration, on_complete=on_complete,
                          pool=_pool, **_worker)
    report = scheduler.run(graph)
    recorder = _recorder.get()
    if recorder is not None:
        for key, placed in report.placed.items():
            recorder.jobs.append(JobRecord(
                repr(key), graph.get(key).kind, placed.path.placement.value,
                tuple(sorted(placed.cores)), placed.node_index, placed.gpu_index,
                report.timings[key]))
    return report


def run_many(parts: dict, *, what: str, mem_cap_bytes: int | None = None,
             vram_cap_bytes: int | None = None) -> tuple[dict, dict]:
    """Run several independent calculations as one graph so they share the machine.

    ``parts`` maps a name to ``(jobs, key)``: one calculation's jobs and the key of the job whose
    result is its answer. Returns ``(values, failures)``: each name's answer, and for each name
    that did not produce one, why. A calculation that fails or cannot fit does not stop the
    others. The graph lists the calculations in ``parts`` order and the scheduler works through a
    graph in order, so earlier ones finish, and free what they hold, before later ones start.
    ``what`` names the batch in the progress line logged as each calculation finishes."""
    answer_of = {(name, key): name for name, (_, key) in parts.items()}
    finished = []

    def on_complete(key):
        if key in answer_of:
            finished.append(key)
            logger.info("%s: %s finished (%d of %d)", what, answer_of[key], len(finished),
                        len(answer_of))

    graph = JobGraph(job for name, (jobs, _) in parts.items() for job in scoped(jobs, name))
    report = _run(graph, mem_cap_bytes, vram_cap_bytes, on_complete)
    values = {name: report.results[key] for key, name in answer_of.items() if key in report.results}
    failures = {error.job_key[0]: str(error) for error in report.infeasible}
    for _, failed, text in report.failures:
        failures[failed[0]] = f"computation failed at {failed[1]!r}: {text}"
    for name in parts:
        if name not in values and name not in failures:
            failures[name] = "the scheduler returned no result"
    return values, failures
