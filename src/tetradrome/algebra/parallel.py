# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Multi-core reduction across independent complexes (engine Phase 5).

Homology splits into independent units -- each quantum grading of a knot, and each knot in
a batch -- so reducing them is embarrassingly parallel. The pure-Python reducers are
GIL-bound, so real parallelism needs processes, not threads; this distributes a batch of
complexes over a process pool and reassembles by key.

Parallelism changes only timing: each complex's homology is independent and deterministic,
so the result is identical to reducing the batch serially (the agreement discipline, design
section 4, applied to concurrency). CPU backends only -- GPU reduction must stay
single-process (many processes contending for one device is not the way to use it), so a
GPU backend is rejected loudly rather than silently degraded.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from multiprocessing import Lock, Pool, Value

from .tiers import f2_homology

_CPU_BACKENDS = ("reference", "bitint", "jit", "packed-cpu")


def _parse_cpulist(text: str) -> list[int]:
    """Parse a Linux cpulist string like '0-3,8,10-11' into a list of CPU ids."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _numa_core_order() -> list[int]:
    """CPU ids interleaved across NUMA nodes, so pool workers spread over sockets rather
    than packing one socket first. Falls back to sequential ids when the NUMA topology is
    not exposed (e.g. a single-node host or a non-sysfs OS)."""
    base = "/sys/devices/system/node"
    try:
        nodes = sorted(
            int(n[4:]) for n in os.listdir(base)
            if n.startswith("node") and n[4:].isdigit()
        )
        per_node = []
        for nd in nodes:
            with open(f"{base}/node{nd}/cpulist") as fh:
                per_node.append(_parse_cpulist(fh.read().strip()))
        order: list[int] = []
        i = 0
        while any(i < len(cpus) for cpus in per_node):
            for cpus in per_node:
                if i < len(cpus):
                    order.append(cpus[i])
            i += 1
        if order:
            return order
    except (OSError, ValueError):
        pass
    return list(range(os.cpu_count() or 1))


def _pin_init(counter, lock, cores):
    with lock:
        idx = counter.value
        counter.value += 1
    os.sched_setaffinity(0, {cores[idx % len(cores)]})


def _worker(args):
    key, cx, backend = args
    return key, f2_homology(cx, backend=backend)


def parallel_f2_homology(
    items, *, backend: str = "bitint", workers: int | None = None, pin: bool = False
) -> dict:
    """F2 homology of many GradedComplexes across processes.

    `items` is a mapping ``{key: complex}`` or an iterable of ``(key, complex)`` pairs;
    the return is ``{key: homology}``, identical to reducing each item serially. `workers`
    defaults to the CPU count; with one worker or fewer than two items the batch is reduced
    in-process (the pool would only add overhead). `pin=True` (Linux only) pins workers to
    CPUs interleaved across NUMA nodes to cut cross-socket memory traffic. CPU backends only.
    """
    if backend not in _CPU_BACKENDS:
        raise ValueError(
            f"parallel_f2_homology is for CPU backends {_CPU_BACKENDS}; got {backend!r}. "
            "GPU reduction must run single-process."
        )
    if pin and not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("pin=True requires Linux (os.sched_setaffinity is unavailable here).")
    pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
    if workers is None:
        workers = os.cpu_count() or 1
    if workers <= 1 or len(pairs) < 2:
        return {key: f2_homology(cx, backend=backend) for key, cx in pairs}
    tasks = [(key, cx, backend) for key, cx in pairs]
    if pin:
        cores = _numa_core_order()
        with Pool(workers, initializer=_pin_init,
                  initargs=(Value("i", 0), Lock(), cores)) as pool:
            results = pool.map(_worker, tasks, chunksize=1)
    else:
        # chunksize=1: complexes vary widely in cost, so fine-grained hand-out balances best.
        with Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks, chunksize=1)
    return dict(results)
