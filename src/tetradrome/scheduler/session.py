# SPDX-License-Identifier: Apache-2.0

"""The engines' way into the scheduler: run one graph on this machine and return one result.

Every engine calculation enters through ``run_to_result``, so one calibration serves the whole
process: the work rates and dispatch overheads one run measures choose the paths of every run
after it. The machine is detected once per process for the same reason a small calculation can
afford to come through here at all: probing the topology costs more than many of them do.
"""
from __future__ import annotations

import dataclasses

from ..errors import TetradromeError
from .executor import Scheduler
from .inventory import Machine, detect_machine
from .routing import Calibration

_calibration = Calibration()
_machine: Machine | None = None


def capped_machine(mem_cap_bytes: int | None, vram_cap_bytes: int | None) -> Machine:
    """The detected machine, optionally lowered to a tighter system-RAM and/or per-GPU VRAM
    ceiling. A cap only ever lowers a limit; it never promises more than the hardware has."""
    global _machine
    if _machine is None:
        _machine = detect_machine()
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
    scheduler = Scheduler(capped_machine(mem_cap_bytes, vram_cap_bytes), calibration=_calibration)
    report = scheduler.run(graph)
    if report.infeasible:
        raise report.infeasible[0]
    if report.failures:
        _, failed, text = report.failures[0]
        raise TetradromeError(f"computation failed at {failed!r}: {text}")
    if key not in report.results:
        raise TetradromeError(f"scheduler returned no result for {key!r}")
    return report.results[key]
