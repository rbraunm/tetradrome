#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Scaling study for the grid (knot Floer) engine.

Runs the whole grid->Poincare graph through the one compute scheduler and reports, per grid, where
the n! generator count's cost lands: the summed worker CPU of each phase -- *generation* (slices
enumerating permutations and computing gradings + differentials), the *merge* (one job folding
every slice into the per-grading complexes), and *reduction* (the F2 linear algebra) -- against the
end-to-end wall, plus a parent-versus-worker-children peak memory split. Wall far above the summed
worker CPU is parent overhead (scheduling and pickling results through the parent, which this graph
does not share via shared memory); a large merge is the single-core fold; a large reduction is the
linear algebra. So the wall (which phase, compute versus parent overhead, time versus memory) is
located empirically rather than guessed. Run it on real hardware and read the curve.

Examples:
    # tabulated knots, increasing grid size, default bitint reducer
    python scripts/bench_grid_floer.py --knots 3_1 4_1 5_1 5_2 8_19

    # synthetic grids to push the generator count
    python scripts/bench_grid_floer.py --sizes 8 9 10

    # study spilling/feasibility under a tighter system-RAM ceiling
    python scripts/bench_grid_floer.py --sizes 9 10 --mem-cap-gib 8

    # compare reducer backends on one knot
    python scripts/bench_grid_floer.py --knots 8_19 --backend reference
    python scripts/bench_grid_floer.py --knots 8_19 --backend packed-gpu
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import threading
import time
from collections import defaultdict

from tetradrome.engines.floer import GridDiagram, staircase_grid
from tetradrome.engines.floer.scheduling import whole_knot_graph
from tetradrome.scheduler import Scheduler, detect_machine


def _time(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


# -- Honest memory measurement (Linux /proc + cgroup) -------------------------------------
# tracemalloc only sees the parent's Python heap; the generation/reduction worker PROCESSES
# are invisible to it, so the old peak undercounted the true footprint at high worker counts.
# Instead sample PSS (proportional set size -- shared pages counted once, so fork/COW pages
# are not double-counted) of the parent and of its worker descendants SEPARATELY, plus the
# cgroup memory.current, on a background thread, keeping per-phase maxima. The parent/children
# split is the point: it separates the parent's accumulated record set (the C(n) term) from
# the worker + IPC footprint (the parallelism term), which a single number conflates.

_CGROUP_CURRENT = ("/sys/fs/cgroup/memory.current",                   # cgroup v2
                   "/sys/fs/cgroup/memory/memory.usage_in_bytes")     # cgroup v1


def _proc_metrics_available() -> bool:
    return os.path.isdir("/proc") and os.path.exists("/proc/self/stat")


def _read_first_int(path: str):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _pss_kib(pid: int):
    """PSS of one process in KiB (smaps_rollup); fall back to RSS; None if the pid is gone."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def _descendants(root: int) -> list:
    """root plus every descendant pid, from a single /proc ppid scan."""
    kids = defaultdict(list)
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                after_comm = f.read().rsplit(") ", 1)[1].split()
            kids[int(after_comm[1])].append(int(entry))   # field after comm: state, ppid, ...
        except (OSError, IndexError, ValueError):
            continue
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(kids.get(pid, []))
    return out


def _cgroup_current():
    for path in _CGROUP_CURRENT:
        value = _read_first_int(path)
        if value is not None:
            return value
    return None


class _TreeSampler:
    """Background sampler of a process tree's PSS + cgroup usage, with resettable maxima."""

    def __init__(self, root_pid: int, interval: float = 0.1):
        self._root = root_pid
        self._interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.reset()

    def reset(self):
        with self._lock:
            self._parent = self._children = self._total = self._cgroup = 0
            self._samples = 0

    def snapshot(self) -> dict:
        with self._lock:
            cg = self._cgroup / 2**20 if self._cgroup else None
            return {"parent": self._parent / 1024.0, "children": self._children / 1024.0,
                    "total": self._total / 1024.0, "cgroup": cg, "samples": self._samples}

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _loop(self):
        while not self._stop.is_set():
            parent = _pss_kib(self._root) or 0
            children = sum(_pss_kib(pid) or 0
                           for pid in _descendants(self._root) if pid != self._root)
            cgroup = _cgroup_current()
            with self._lock:
                self._parent = max(self._parent, parent)
                self._children = max(self._children, children)
                self._total = max(self._total, parent + children)
                if cgroup is not None:
                    self._cgroup = max(self._cgroup, cgroup)
                self._samples += 1
            self._stop.wait(self._interval)


_PHASE = {"gen_slice": "generation", "gen_merge": "merge", "reduce": "reduction",
          "assemble": "assemble"}


def _capped(machine, mem_cap_bytes):
    # Lower the scheduler's system-RAM ceiling for the run; never above the detected hardware.
    if mem_cap_bytes is None:
        return machine
    return dataclasses.replace(machine, mem_cap_bytes=min(machine.mem_cap_bytes, mem_cap_bytes))


def measure(grid, *, backend, mem_cap_bytes=None, interval=0.1):
    """Run the whole grid->Poincare graph through the scheduler; return the end-to-end wall, the
    summed worker CPU per phase (generation / merge / reduction) from the RunReport, homology
    support size, the residence + spill counters, and a parent-vs-worker-children peak PSS
    breakdown (MiB). Fails loud on an infeasible or failed run rather than reporting a partial."""
    sampler = _TreeSampler(os.getpid(), interval) if _proc_metrics_available() else None
    if sampler:
        sampler.start()
        sampler.reset()
    machine = _capped(detect_machine(), mem_cap_bytes)
    graph, key = whole_knot_graph(grid, backend=backend)
    report, wall = _time(lambda: Scheduler(machine).run(graph))
    mem = sampler.snapshot() if sampler else None
    if sampler:
        sampler.stop()
    if report.infeasible:
        raise RuntimeError(f"infeasible on this machine: {report.infeasible[0]}")
    if report.failures:
        raise RuntimeError(f"scheduler failed: {report.failures[0][2]}")
    cpu = defaultdict(float)
    for job_key, seconds in report.timings.items():
        cpu[_PHASE.get(job_key[0], job_key[0])] += seconds
    return {"wall": wall, "count": math.factorial(grid.n), "support": len(report.results[key]),
            "gen_s": cpu.get("generation", 0.0), "merge_s": cpu.get("merge", 0.0),
            "red_s": cpu.get("reduction", 0.0),
            "shared": report.shared_count, "spilled": report.spill_count, "mem": mem}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knots", nargs="*", default=[],
                        help="KnotInfo names (representative reduction cost)")
    parser.add_argument("--sizes", nargs="*", type=int, default=[],
                        help="synthetic grid sizes (pushes the generator count)")
    parser.add_argument("--backend", default="bitint",
                        help="reduction backend: reference|bitint|jit|packed-cpu|packed-gpu")
    parser.add_argument("--mem-cap-gib", type=float, default=0.0,
                        help="cap the scheduler's system-RAM ceiling at this many GiB (0 = the "
                             "detected machine). Lower it to study spilling and feasibility; a grid "
                             "whose smallest unavoidable working set exceeds the cap is reported "
                             "infeasible rather than OOMing.")
    args = parser.parse_args()

    targets = ([(name, GridDiagram.from_knotinfo(name)) for name in args.knots] +
               [(f"staircase-{n}", staircase_grid(n)) for n in args.sizes])
    if not targets:
        parser.error("give --knots and/or --sizes")

    mem_cap_bytes = int(args.mem_cap_gib * 2**30) if args.mem_cap_gib else None
    cap_label = f"{args.mem_cap_gib:.0f} GiB" if mem_cap_bytes else "machine"
    print(f"backend={args.backend}  mem-cap={cap_label}  "
          f"(gen/merge/reduce are summed worker CPU; wall is end-to-end)")
    if _proc_metrics_available():
        print("memory (MiB): peak PSS -- Par=parent process, Chd=summed worker children; "
              "cgPk=peak cgroup memory.current")
    else:
        print("memory: UNAVAILABLE on this OS (needs Linux /proc + cgroup) -- shown as n/a")
    print()
    header = (f"{'target':<13}{'n':>3}{'n!':>13}{'wall':>9}{'gen':>8}{'merge':>8}{'reduce':>8}"
              f"{'shr':>4}{'spl':>4}{'parPk':>9}{'chdPk':>9}{'cgPk':>9}{'(M,A)':>7}")
    print(header)
    print("-" * len(header))

    def _c(value):
        return f"{value:>9.1f}" if value is not None else f"{'n/a':>9}"

    for name, grid in targets:
        try:
            r = measure(grid, backend=args.backend, mem_cap_bytes=mem_cap_bytes)
        except RuntimeError as exc:
            print(f"{name:<13}{grid.n:>3}{math.factorial(grid.n):>13,}   {exc}")
            continue
        m = r["mem"]
        if m is None:
            mem_cells = _c(None) + _c(None) + f"{'n/a':>9}"
        else:
            cg = m["cgroup"]
            mem_cells = (_c(m["parent"]) + _c(m["children"])
                         + (f"{cg:>9.1f}" if cg is not None else f"{'n/a':>9}"))
        print(f"{name:<13}{grid.n:>3}{r['count']:>13,}{r['wall']:>9.3f}"
              f"{r['gen_s']:>8.2f}{r['merge_s']:>8.2f}{r['red_s']:>8.2f}"
              f"{r['shared']:>4}{r['spilled']:>4}{mem_cells}{r['support']:>7}")


if __name__ == "__main__":
    main()
