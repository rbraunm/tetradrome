# SPDX-License-Identifier: Apache-2.0

"""Measure this machine once, so every later calculation starts out knowing what its jobs cost.

    python -m tetradrome.calibrate

The benchmark runs each engine's small representative calculations through the scheduler and
records, for this machine and this software:

- the work rate of every kind of job (seconds per cost unit);
- what each worker placement costs to dispatch;
- what a fresh worker costs to start (its RAM baseline and a GPU context's VRAM).

The figures are kept by ``scheduler.calibration_store`` and are measured again only when the
machine's processors or GPUs change, when the code or its numerical libraries change, or when this
command is run. The first calculation on a machine with no current figures runs the benchmark
itself, and says so.
"""
from __future__ import annotations

import importlib
import logging
import statistics
import sys
import time
from collections import defaultdict

from .scheduler import (
    Calibration,
    ComputePath,
    Job,
    JobGraph,
    Placement,
    Scheduler,
    calibration_store,
    detect_machine,
)

logger = logging.getLogger(__name__)

_DISPATCH_SAMPLES = 8
# Other work on the machine can only slow a pass down, so the fastest of a few is the estimate.
_PASSES = 3


def _nothing(inputs, deps):
    return None


def _engines():
    """Each engine's ``benchmark(run)``: it runs that engine's representative calculations by
    calling ``run(graph, key)``, which returns the result stored under ``key``."""
    # By module path: the invariants package exports a function named like its compute module.
    return tuple(importlib.import_module(module).benchmark
                 for module in ("tetradrome.invariants.compute",
                                "tetradrome.engines.floer.scheduling"))


def _dispatch_seconds(machine, placement, worker) -> float:
    """The median seconds to hand a job with nothing to do to ``placement`` and reap it."""
    samples = []
    for index in range(_DISPATCH_SAMPLES):
        calibration = Calibration()
        job = Job(key=("dispatch", index), run=_nothing, inputs={},
                  paths=(ComputePath(placement, cores=1, ram_bytes=1),))
        report = Scheduler(machine, calibration=calibration, **worker).run(JobGraph([job]))
        if report.failures or report.infeasible:
            raise RuntimeError(f"dispatch benchmark failed on {placement.value}: "
                               f"{report.failures or report.infeasible}")
        samples.append(calibration.overhead(placement))
    return statistics.median(samples)


def measure(machine) -> dict:
    """Run the benchmark on ``machine`` and return its figures, in the stored form."""
    ram_baseline, context_vram = Scheduler(machine).worker_overhead()
    worker = {"worker_ram_baseline": ram_baseline, "context_vram_reserve": context_vram}
    rates: dict = {}
    for _ in range(_PASSES):
        seconds: dict = defaultdict(float)
        units: dict = defaultdict(float)

        def run(graph, key):
            report = Scheduler(machine, calibration=Calibration(), **worker).run(graph)
            if report.infeasible:
                raise report.infeasible[0]
            if report.failures:
                raise RuntimeError(f"benchmark calculation failed: {report.failures[0][2]}")
            for job_key, job_seconds in report.timings.items():
                job = graph.get(job_key)
                if job.cost > 0:
                    seconds[job.kind] += job_seconds
                    units[job.kind] += float(job.cost)
            return report.results[key]

        for benchmark in _engines():
            benchmark(run)
        for kind in units:
            rate = seconds[kind] / units[kind]
            rates[kind] = min(rate, rates.get(kind, rate))
    return {
        "rates": [{"kind": kind, "device": "cpu", "seconds_per_unit": rates[kind]}
                  for kind in sorted(rates)],
        "overheads": [{"placement": placement.name,
                       "seconds": _dispatch_seconds(machine, placement, worker)}
                      for placement in (Placement.CPU_PINNED, Placement.CPU_UNPINNED)],
        "worker": {"ram_baseline": ram_baseline, "context_vram": context_vram},
    }


def ensure(calibration: Calibration, machine) -> dict:
    """Give ``calibration`` this machine's figures, measuring them first if none are current.
    Returns the figures."""
    directory = calibration_store.store_directory()
    figures, why_not = calibration_store.read(machine, directory)
    if figures is None:
        logger.warning("Measuring how fast this machine runs each kind of job, because %s. This "
                       "is done once and kept; `python -m tetradrome.calibrate` repeats it.",
                       why_not)
        started = time.perf_counter()
        figures = measure(machine)
        path = calibration_store.write(machine, directory, figures)
        logger.warning("Measured in %.0f s; figures kept in %s", time.perf_counter() - started,
                       path)
    calibration.fix(figures)
    return figures


def describe(figures: dict) -> str:
    lines = [f"  {entry['kind']:<36} {entry['seconds_per_unit']:.3g} s per unit on {entry['device']}"
             for entry in figures["rates"]]
    lines += [f"  dispatch to {entry['placement']:<24} {entry['seconds'] * 1000:.1f} ms"
              for entry in figures["overheads"]]
    lines.append(f"  fresh worker RAM baseline            {figures['worker']['ram_baseline']} bytes")
    lines.append(f"  GPU context VRAM                     {figures['worker']['context_vram']} bytes")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    machine = detect_machine()
    print(machine.describe())
    started = time.perf_counter()
    figures = measure(machine)
    path = calibration_store.write(machine, calibration_store.store_directory(), figures)
    print(f"measured in {time.perf_counter() - started:.0f} s; kept in {path}")
    print(describe(figures))
    return 0


if __name__ == "__main__":
    # Jobs name their functions by module, and a worker cannot import "__main__".
    from tetradrome.calibrate import main as _main
    sys.exit(_main())
