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

import dataclasses
import logging

from ..errors import TetradromeError
from .executor import Scheduler
from .graph import JobGraph
from .inventory import Machine, detect_machine
from .job import scoped
from .routing import Calibration

logger = logging.getLogger(__name__)

_calibration = Calibration()
_machine: Machine | None = None
_worker: dict = {}


def capped_machine(mem_cap_bytes: int | None, vram_cap_bytes: int | None) -> Machine:
    """The detected machine, optionally lowered to a tighter system-RAM and/or per-GPU VRAM
    ceiling. A cap only ever lowers a limit; it never promises more than the hardware has."""
    global _machine
    if _machine is None:
        from ..calibrate import ensure          # the benchmark imports the engines, which import this
        machine = detect_machine()
        figures = ensure(_calibration, machine)
        _worker.update(worker_ram_baseline=figures["worker"]["ram_baseline"],
                       context_vram_reserve=figures["worker"]["context_vram"])
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
    scheduler = Scheduler(machine, calibration=_calibration, on_complete=on_complete, **_worker)
    return scheduler.run(graph)


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
