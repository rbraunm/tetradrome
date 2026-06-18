#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Scaling study for the grid (knot Floer) engine.

Measures, per grid, the two costs that the n! generator count splits into -- *generation*
(enumerating permutations and computing gradings + differentials) and *reduction* (the F2
linear algebra) -- plus peak traced memory, so the wall (time vs memory, generation vs
reduction) is located empirically rather than guessed. Run it on real hardware and read the
curve.

Examples:
    # tabulated knots, increasing grid size, default bitint reducer, serial
    python scripts/bench_grid_floer.py --knots 3_1 4_1 5_1 5_2 8_19

    # synthetic grids to push the generator count, with parallel generation + reduction
    python scripts/bench_grid_floer.py --sizes 5 6 7 8 9 --gen-workers 16 --workers 16

    # compare reducer backends on one knot
    python scripts/bench_grid_floer.py --knots 8_19 --backend reference
    python scripts/bench_grid_floer.py --knots 8_19 --backend packed-gpu
"""
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import defaultdict

from tetradrome.engines.floer import (
    GridDiagram,
    dense_reduction_bytes,
    grading_histogram,
    grid_complexes,
    parallel_grid_complexes,
    reduce_complexes,
    staircase_grid,
)


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


def measure(grid, *, backend, workers, pin, gen_workers, interval=0.1):
    """Run generation then reduction; return timings, homology support size, and a per-phase
    memory breakdown (parent PSS vs worker-children PSS vs cgroup peak), in MiB."""
    sampler = _TreeSampler(os.getpid(), interval) if _proc_metrics_available() else None
    if sampler:
        sampler.start()
    try:
        if sampler:
            sampler.reset()
        if gen_workers > 1:
            complexes, gen_s = _time(lambda: parallel_grid_complexes(grid, gen_workers))
        else:
            complexes, gen_s = _time(lambda: grid_complexes(grid))
        gen_mem = sampler.snapshot() if sampler else None

        if sampler:
            sampler.reset()
        poincare, red_s = _time(
            lambda: reduce_complexes(complexes, backend=backend, workers=workers, pin=pin)
        )
        red_mem = sampler.snapshot() if sampler else None
    finally:
        if sampler:
            sampler.stop()

    mem = None if sampler is None else {"gen": gen_mem, "red": red_mem}
    return {"gen_s": gen_s, "red_s": red_s, "count": math.factorial(grid.n),
            "support": len(poincare), "mem": mem}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knots", nargs="*", default=[],
                        help="KnotInfo names (representative reduction cost)")
    parser.add_argument("--sizes", nargs="*", type=int, default=[],
                        help="synthetic grid sizes (isolates generation scaling)")
    parser.add_argument("--backend", default="bitint",
                        help="reduction backend: reference|bitint|jit|packed-cpu|packed-gpu")
    parser.add_argument("--workers", type=int, default=1, help="reduction worker processes")
    parser.add_argument("--gen-workers", type=int, default=1, help="generation worker processes")
    parser.add_argument("--pin", action="store_true", help="NUMA-pin reduction workers (Linux)")
    parser.add_argument("--mem-budget-gib", type=float, default=0.0,
                        help="skip any size whose projected peak dense reduction memory exceeds "
                             "this many GiB (0 = no guard); refuses up front instead of OOMing "
                             "mid-sweep. The projection is the exact dense-matrix bound from the "
                             "grading dimensions (gradings-only histogram, memory-safe).")
    args = parser.parse_args()

    if args.pin and not hasattr(os, "sched_setaffinity"):
        parser.error(
            "--pin is Linux-only (NUMA affinity via os.sched_setaffinity); this OS lacks it. "
            "Drop --pin here, or run the parallel/pinned sweep on the Linux cluster (where "
            "fork also avoids Windows' spawn overhead). The GPU-backend comparison is fine here."
        )

    targets = ([(name, GridDiagram.from_knotinfo(name)) for name in args.knots] +
               [(f"staircase-{n}", staircase_grid(n)) for n in args.sizes])
    if not targets:
        parser.error("give --knots and/or --sizes")

    print(f"backend={args.backend}  reduction-workers={args.workers}  "
          f"gen-workers={args.gen_workers}  pin={args.pin}")
    if _proc_metrics_available():
        print("memory (MiB): PSS per phase -- Par=parent process, Chd=summed worker children; "
              "cgPk=peak cgroup memory.current")
    else:
        print("memory: UNAVAILABLE on this OS (needs Linux /proc + cgroup) -- shown as n/a")
    print()
    header = (f"{'target':<13}{'n':>3}{'n!':>13}{'gen(s)':>9}{'red(s)':>9}"
              f"{'genPar':>8}{'genChd':>8}{'redPar':>8}{'redChd':>8}{'cgPk':>9}{'(M,A)':>7}")
    print(header)
    print("-" * len(header))

    def _c(value):
        return f"{value:>8.1f}" if value is not None else f"{'n/a':>8}"

    for name, grid in targets:
        if args.mem_budget_gib:
            projected_gib = dense_reduction_bytes(
                grading_histogram(grid, args.gen_workers)) / 2**30
            if projected_gib > args.mem_budget_gib:
                print(f"{name:<13}{grid.n:>3}{math.factorial(grid.n):>13,}"
                      f"   skipped: ~{projected_gib:.1f} GiB projected dense reduction > "
                      f"{args.mem_budget_gib:.1f} GiB budget")
                continue
        r = measure(grid, backend=args.backend, workers=args.workers, pin=args.pin,
                    gen_workers=args.gen_workers)
        m = r["mem"]
        if m is None:
            mem_cells = _c(None) * 4 + f"{'n/a':>9}"
        else:
            cg = max((p for p in (m["gen"]["cgroup"], m["red"]["cgroup"]) if p is not None),
                     default=None)
            mem_cells = (_c(m["gen"]["parent"]) + _c(m["gen"]["children"])
                         + _c(m["red"]["parent"]) + _c(m["red"]["children"])
                         + (f"{cg:>9.1f}" if cg is not None else f"{'n/a':>9}"))
        print(f"{name:<13}{grid.n:>3}{r['count']:>13,}{r['gen_s']:>9.3f}{r['red_s']:>9.3f}"
              f"{mem_cells}{r['support']:>7}")


if __name__ == "__main__":
    main()
