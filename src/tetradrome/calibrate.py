# SPDX-License-Identifier: Apache-2.0

"""Measure this machine once, so every later calculation starts out knowing what its jobs cost.

    python -m tetradrome.calibrate

The benchmark runs each engine's small representative calculations and records, for this machine
and this software:

- what every kind of job costs: a fixed part per job and seconds per cost unit;
- how much slower each kind runs when two jobs share a physical core's two hyperthreads, which
  decides whether the scheduler uses a core's second thread for that kind;
- what each worker placement costs to dispatch to a waiting worker, how much of that is the
  scheduler's own time, and what starting a worker adds;
- what a fresh worker holds before it does any work (its RAM baseline and a GPU context's VRAM).

The figures are kept by ``scheduler.calibration_store`` and are measured again only when the
machine's processors or GPUs change, when the code or its numerical libraries change, or when this
command is run. The first calculation on a machine with no current figures runs the benchmark
itself, and says so.
"""
from __future__ import annotations

import importlib
import logging
import math
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
    Shard,
    calibration_store,
    detect_machine,
)
from .scheduler.executor import _start_context
from .scheduler.hostplatform import for_host

logger = logging.getLogger(__name__)

_DISPATCH_SAMPLES = 8
_HANDOVER_JOBS = 64
# Other work on the machine can only slow a pass down, so the fastest of a few is the estimate.
_PASSES = 3
# A shared-core measurement repeats its job until it has run about this long, so that two workers
# released together stay on the core together.
_SHARED_SECONDS = 0.25
_SHARED_REPEATS_MOST = 10000


def _nothing(inputs, deps):
    return None


def _engines():
    """Each engine's ``benchmark(run)``: it runs that engine's representative calculations by
    calling ``run(graph, key)``, which returns the result stored under ``key``."""
    # By module path: the invariants package exports a function named like its compute module.
    return tuple(importlib.import_module(module).benchmark
                 for module in ("tetradrome.invariants.compute",
                                "tetradrome.engines.floer.scheduling"))


def fit(points) -> tuple:
    """``(fixed, per_unit)`` seconds for one kind from its jobs' ``(cost, seconds)``: the line
    that predicts each job's time with the least relative error, so small jobs count as much as
    large ones. A kind whose jobs are all one size, or whose line would give a negative part, is
    given no fixed part and its total seconds over its total cost."""
    weight = [1.0 / max(seconds, 1e-9) ** 2 for _, seconds in points]
    total = sum(weight)
    cost_sum = sum(w * cost for w, (cost, _) in zip(weight, points))
    time_sum = sum(w * seconds for w, (_, seconds) in zip(weight, points))
    spread = total * sum(w * cost * cost for w, (cost, _) in zip(weight, points)) - cost_sum ** 2
    if spread > 1e-9 * cost_sum ** 2:           # the jobs are not all one size
        per_unit = (total * sum(w * cost * seconds for w, (cost, seconds) in zip(weight, points))
                    - cost_sum * time_sum) / spread
        fixed = (time_sum - per_unit * cost_sum) / total
        if per_unit >= 0 and fixed >= 0:
            return fixed, per_unit
    return 0.0, sum(seconds for _, seconds in points) / sum(cost for cost, _ in points)


def _work_rates(machine, worker) -> dict:
    """``(fixed, per_unit)`` seconds of every kind the engines' benchmarks run through the
    scheduler on ``machine``, fitted to each job's fastest time over ``_PASSES``. A Scheduler
    with nothing measured gives every job a physical core of its own."""
    fastest: dict = defaultdict(dict)       # kind -> {(which graph, job key): (cost, seconds)}
    for _ in range(_PASSES):
        graphs = iter(range(10 ** 9))

        def run(graph, key):
            which = next(graphs)
            report = Scheduler(machine, calibration=Calibration(), **worker).run(graph)
            if report.infeasible:
                raise report.infeasible[0]
            if report.failures:
                raise RuntimeError(f"benchmark calculation failed: {report.failures[0][2]}")
            for job_key, job_seconds in report.timings.items():
                job = graph.get(job_key)
                if job.cost > 0:
                    seen = fastest[job.kind].get((which, job_key), (0.0, job_seconds))[1]
                    fastest[job.kind][(which, job_key)] = (float(job.cost), min(seen, job_seconds))
            return report.results[key]

        for benchmark in _engines():
            benchmark(run)
    return {kind: fit(list(points.values())) for kind, points in fastest.items()}


def _samples() -> dict:
    """For each kind, the costliest of its jobs in the engines' benchmark calculations, as
    ``(cost, run, inputs, deps)``. Found by running those calculations in this process, each job
    after the ones it depends on."""
    samples: dict = {}

    def run(graph, key):
        results: dict = {}
        while len(results) < len(graph):
            for job in graph.ready(results):
                deps = {dep: (results[dep.producer][dep.key] if isinstance(dep, Shard)
                              else results[dep])
                        for dep in job.dependencies}
                if job.cost > samples.get(job.kind, (0.0,))[0]:
                    samples[job.kind] = (job.cost, job.run, job.inputs, deps)
                results[job.key] = job.run(job.inputs, deps)
        return results[key]

    for benchmark in _engines():
        benchmark(run)
    return samples


def _repeat_main(run, inputs, deps, cpu, repeats, barrier, done):
    try:
        for_host().pin({cpu})
        barrier.wait(timeout=120)
        started = time.perf_counter()
        for _ in range(repeats):
            run(inputs, deps)
        done.put((time.perf_counter() - started) / repeats)
    except Exception as error:
        done.put(repr(error))


def _seconds_on(cpus, run, inputs, deps, repeats) -> float:
    """Mean seconds per repeat of one job when a worker pinned to each of ``cpus`` runs it
    ``repeats`` times, all released together."""
    context = _start_context()
    barrier = context.Barrier(len(cpus))
    done = context.Queue()
    workers = [context.Process(target=_repeat_main,
                               args=(run, inputs, deps, cpu, repeats, barrier, done))
               for cpu in cpus]
    for process in workers:
        process.start()
    try:
        reported = [done.get(timeout=600) for _ in workers]
    finally:
        for process in workers:
            process.join()
    failed = [item for item in reported if isinstance(item, str)]
    if failed:
        raise RuntimeError(f"shared-core benchmark worker failed: {failed[0]}")
    return statistics.mean(reported)


def _sibling_factors(machine, rates) -> list:
    """For each kind, its time with a physical core's two threads both running it, over its time
    with the core to itself. Both are measured on one core while the rest of the machine is idle,
    since a processor runs each core faster the fewer of them are working. Empty when no core here
    has two threads the scheduler may use."""
    pair = next((sorted(core)[:2] for node in machine.nodes for core in node.physical_cores
                 if len(core) >= 2), None)
    if pair is None:
        return []
    factors = []
    for kind, (cost, run, inputs, deps) in sorted(_samples().items()):
        fixed, per_unit = rates[kind]
        repeats = min(max(1, math.ceil(_SHARED_SECONDS / max(fixed + cost * per_unit, 1e-9))),
                      _SHARED_REPEATS_MOST)
        alone = _seconds_on(pair[:1], run, inputs, deps, repeats)
        shared = _seconds_on(pair, run, inputs, deps, repeats)
        factors.append({"kind": kind, "factor": shared / alone})
    return factors


def _dispatch_seconds(machine, placement, worker) -> tuple:
    """``(start, dispatch)`` for ``placement``: the seconds that starting a worker adds to a
    job's dispatch when none is waiting, and the median seconds to hand a job with nothing to do
    to a waiting worker and reap it. From one run of such jobs in a chain: the first starts the
    worker and the rest reuse it."""
    jobs = [Job(key=("dispatch", index), run=_nothing, inputs={},
                paths=(ComputePath(placement, cores=1, ram_bytes=1),),
                dependencies=[("dispatch", index - 1)] if index else [])
            for index in range(_DISPATCH_SAMPLES + 1)]
    report = Scheduler(machine, calibration=Calibration(), **worker).run(JobGraph(jobs))
    if report.failures or report.infeasible:
        raise RuntimeError(f"dispatch benchmark failed on {placement.value}: "
                           f"{report.failures or report.infeasible}")
    first, *later = (report.overheads[job.key] for job in jobs)
    dispatch = statistics.median(later)
    return max(first - dispatch, 0.0), dispatch


def _handover_seconds(machine, placement, worker) -> float:
    """The scheduler's own CPU seconds per job when it hands a burst of jobs with nothing to do
    to ``placement`` and takes their results back."""
    jobs = [Job(key=("handover", index), run=_nothing, inputs={},
                paths=(ComputePath(placement, cores=1, ram_bytes=1),))
            for index in range(_HANDOVER_JOBS)]
    started = time.process_time()
    report = Scheduler(machine, calibration=Calibration(), **worker).run(JobGraph(jobs))
    spent = time.process_time() - started
    if report.failures or report.infeasible:
        raise RuntimeError(f"hand-over benchmark failed on {placement.value}: "
                           f"{report.failures or report.infeasible}")
    return spent / len(jobs)


def measure(machine) -> dict:
    """Run the benchmark on ``machine`` and return its figures, in the stored form."""
    ram_baseline, context_vram = Scheduler(machine).worker_overhead()
    worker = {"worker_ram_baseline": ram_baseline, "context_vram_reserve": context_vram}
    rates = _work_rates(machine, worker)
    dispatch = {placement: _dispatch_seconds(machine, placement, worker)
                for placement in (Placement.CPU_PINNED, Placement.CPU_UNPINNED)}
    return {
        "rates": [{"kind": kind, "device": "cpu", "seconds_fixed": rates[kind][0],
                   "seconds_per_unit": rates[kind][1]} for kind in sorted(rates)],
        "sibling": _sibling_factors(machine, rates),
        "overheads": [{"placement": placement.name, "seconds": seconds,
                       "scheduler_seconds": _handover_seconds(machine, placement, worker)}
                      for placement, (_, seconds) in dispatch.items()],
        "worker": {"ram_baseline": ram_baseline, "context_vram": context_vram,
                   "start_seconds": dispatch[Placement.CPU_PINNED][0]},
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
        logger.info("%s", describe(figures))
    calibration.fix(figures)
    return figures


def describe(figures: dict) -> str:
    calibration = Calibration()
    calibration.fix(figures)
    factor = {entry["kind"]: entry["factor"] for entry in figures["sibling"]}
    lines = []
    for entry in figures["rates"]:
        line = (f"  {entry['kind']:<36} {entry['seconds_fixed'] * 1000:.2f} ms + "
                f"{entry['seconds_per_unit']:.3g} s per unit on {entry['device']}")
        if entry["kind"] in factor:
            use = "used" if calibration.shares(entry["kind"]) else "not used"
            line += f"; x{factor[entry['kind']]:.2f} sharing a core, second thread {use}"
        lines.append(line)
    lines += [f"  dispatch to {entry['placement']:<24} {entry['seconds'] * 1000:.1f} ms, of which "
              f"the scheduler's own {entry['scheduler_seconds'] * 1000:.2f} ms"
              for entry in figures["overheads"]]
    lines.append(f"  starting a worker                    "
                 f"{figures['worker']['start_seconds'] * 1000:.1f} ms")
    worker = figures["worker"]
    lines.append(f"  fresh worker RAM baseline            {worker['ram_baseline']} bytes")
    lines.append(f"  GPU context VRAM                     {worker['context_vram']} bytes")
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
