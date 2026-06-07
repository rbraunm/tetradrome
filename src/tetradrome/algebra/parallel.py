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
from collections.abc import Iterable, Mapping
from multiprocessing import Pool

from .tiers import f2_homology

_CPU_BACKENDS = ("reference", "bitint", "packed-cpu")


def _worker(args):
    key, cx, backend = args
    return key, f2_homology(cx, backend=backend)


def parallel_f2_homology(items, *, backend: str = "bitint", workers: int | None = None) -> dict:
    """F2 homology of many GradedComplexes across processes.

    `items` is a mapping ``{key: complex}`` or an iterable of ``(key, complex)`` pairs;
    the return is ``{key: homology}``, identical to reducing each item serially. `workers`
    defaults to the CPU count; with one worker or fewer than two items the batch is reduced
    in-process (the pool would only add overhead). CPU backends only.
    """
    if backend not in _CPU_BACKENDS:
        raise ValueError(
            f"parallel_f2_homology is for CPU backends {_CPU_BACKENDS}; got {backend!r}. "
            "GPU reduction must run single-process."
        )
    pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
    if workers is None:
        workers = os.cpu_count() or 1
    if workers <= 1 or len(pairs) < 2:
        return {key: f2_homology(cx, backend=backend) for key, cx in pairs}
    tasks = [(key, cx, backend) for key, cx in pairs]
    with Pool(processes=workers) as pool:
        # chunksize=1: complexes vary widely in cost, so fine-grained hand-out balances best.
        results = pool.map(_worker, tasks, chunksize=1)
    return dict(results)
