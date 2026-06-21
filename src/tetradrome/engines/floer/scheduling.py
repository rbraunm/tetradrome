# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Adapter: express a knot's F2 reduction as scheduler jobs.

A grading's reduction is f2_homology on one complex, and every backend returns the identical
answer by the agreement discipline, so a reduction's compute paths are a pure performance
choice: a GPU path when a CUDA reducer is present, and a single-core CPU path that is always
there. The scheduler runs the fastest the machine can serve. Each path is priced by the cost
model's packed peak. The assembly fans the per-grading homologies into the bigraded Poincare
count, the same arithmetic the in-process reducer used.

This builds only the reduction-plus-assembly subgraph. The whole-knot builder prepends
generation and reuses it.

A reduction's run resolves its backend from its input: a forced backend name reduces with that
backend (this is how the agreement tests pin each tier), and ``auto`` lets the worker's
environment decide, so a worker the scheduler placed on a GPU sees the device and one placed on
CPU does not.
"""
from __future__ import annotations

from collections import defaultdict

from ...algebra import available_f2_backends, f2_homology, predict_size
from ...scheduler import ComputePath, Job, JobGraph, Placement

_REDUCE = "reduce"
_ASSEMBLE = "assemble"


def _reduce_run(inputs, deps):
    return f2_homology(inputs["complex"], inputs["backend"])


def _assemble_run(inputs, deps):
    poincare: dict = defaultdict(int)
    for key, homology in deps.items():
        alexander = key[1]                       # key is (_REDUCE, alexander)
        for degree, dimension in homology.items():
            poincare[(-degree, alexander)] += dimension
    return {key: value for key, value in poincare.items() if value}


def _reduction_paths(cx, backend: str) -> tuple:
    peak = predict_size(cx).packed_peak_bytes
    if backend == "packed-gpu":
        return (ComputePath(Placement.GPU, cores=1, ram_bytes=peak, vram_bytes=max(peak, 1)),)
    if backend != "auto":
        return (ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=peak),)
    available = {name: ok for name, ok, _ in available_f2_backends()}
    paths = []
    if available.get("packed-gpu"):
        paths.append(ComputePath(Placement.GPU, cores=1, ram_bytes=peak, vram_bytes=max(peak, 1)))
    paths.append(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=peak))
    return tuple(paths)


def reduction_jobs(complexes: dict, *, backend: str) -> tuple:
    """Reduction jobs (one per grading) plus the assembly that consumes them, as
    ``(jobs, assemble_key)``. The assembly depends on every reduction, so a failed reduction
    abandons the whole knot."""
    jobs = []
    reduce_keys = []
    for alexander, cx in complexes.items():
        key = (_REDUCE, alexander)
        jobs.append(Job(
            key=key, run=_reduce_run,
            inputs={"complex": cx, "backend": backend},
            paths=_reduction_paths(cx, backend),
        ))
        reduce_keys.append(key)
    assemble_key = (_ASSEMBLE,)
    jobs.append(Job(
        key=assemble_key, run=_assemble_run, inputs={},
        paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=0),),
        dependencies=reduce_keys,
    ))
    return jobs, assemble_key


def reduction_graph(complexes: dict, *, backend: str):
    """A JobGraph for reducing ``{A: GradedComplex}`` plus its assembly, and the assembly key
    whose result is the ``{(Maslov, Alexander): dimension}`` Poincare count."""
    jobs, assemble_key = reduction_jobs(complexes, backend=backend)
    return JobGraph(jobs), assemble_key
