# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Grid homology, reduced to knot Floer homology HFK-hat (Phase 6).

The differential preserves the Alexander grading and lowers Maslov by one, so each Alexander
grading is a Maslov-graded F2 complex; we hand it to the shared back end with degree set to
-Maslov, so the back end's degree-raising differential matches. The grid (hat) homology of an
n x n diagram is HFK-hat(K) (x) V^{(n-1)} with V = F2_{(0,0)} (+) F2_{(-1,-1)}, so dividing
the grid Poincare polynomial by (1 + q^{-1} t^{-1})^{n-1} recovers HFK-hat; the quotient is
checked by reconstructing the product (fail loud on mismatch).

The flat KnotInfo marker list does not record which markers are O and which are X, so the
diagram fixes a chirality only up to the global O<->X swap; HFK-hat is therefore determined
up to mirror, (M, A) <-> (-M, -A). The tau invariant (later) pins chirality. The Seifert
genus is the top Alexander grading carrying nonzero HFK-hat (the genus-detection theorem).

Generation and reduction both run through the one scheduler: it spawns the work, holds the
memory model, and reproduces the serial reference exactly. ``mem_cap_bytes`` and ``vram_cap_bytes``
lower the system-RAM and VRAM ceilings the scheduler runs under, never above the detected
hardware, so a computation can be held to a tighter budget than the machine.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict

from ...errors import TetradromeError, UnvalidatedResult
from ...scheduler import Scheduler, detect_machine
from .scheduling import reduction_graph, whole_knot_graph


def _capped_machine(mem_cap_bytes: int | None, vram_cap_bytes: int | None):
    # The detected machine, optionally lowered to a tighter system-RAM and/or per-GPU VRAM ceiling.
    # A cap only ever lowers a limit; it never promises more than the hardware has.
    machine = detect_machine()
    if mem_cap_bytes is None and vram_cap_bytes is None:
        return machine
    capped_ram = (machine.mem_cap_bytes if mem_cap_bytes is None
                  else min(machine.mem_cap_bytes, mem_cap_bytes))
    gpus = machine.gpus
    if vram_cap_bytes is not None:
        gpus = tuple(dataclasses.replace(gpu, vram_bytes=min(gpu.vram_bytes, vram_cap_bytes))
                     for gpu in machine.gpus)
    return dataclasses.replace(machine, mem_cap_bytes=capped_ram, gpus=gpus)


def _run_to_result(graph, key, *, mem_cap_bytes, vram_cap_bytes):
    # Run a graph on the (optionally capped) machine and return its single result, failing loud:
    # an infeasible job, a failed component, or a missing result all raise rather than hand back a
    # partial or silently-empty answer.
    report = Scheduler(_capped_machine(mem_cap_bytes, vram_cap_bytes)).run(graph)
    if report.infeasible:
        raise report.infeasible[0]
    if report.failures:
        _, failed, text = report.failures[0]
        raise TetradromeError(f"computation failed at {failed!r}: {text}")
    if key not in report.results:
        raise TetradromeError(f"scheduler returned no result for {key!r}")
    return report.results[key]


def reduce_complexes(complexes: dict, *, backend: str = "bitint",
                     mem_cap_bytes: int | None = None, vram_cap_bytes: int | None = None) -> dict:
    """Reduce ``{A: GradedComplex}`` to ``{(Maslov, Alexander): dimension}`` through the scheduler.

    The per-grading complexes are independent, so the scheduler reduces them concurrently within
    the machine's budget (spilling rather than failing under memory pressure). Every backend
    returns the identical answer as the reference (the agreement discipline).
    """
    graph, assemble_key = reduction_graph(complexes, backend=backend)
    return _run_to_result(graph, assemble_key,
                          mem_cap_bytes=mem_cap_bytes, vram_cap_bytes=vram_cap_bytes)


def grid_poincare(grid, *, backend: str = "bitint",
                  mem_cap_bytes: int | None = None, vram_cap_bytes: int | None = None) -> dict:
    """Grid (hat) homology as ``{(Maslov, Alexander): dimension}`` over F2, computed end to end
    through the scheduler: generation, a merge partitioned by Alexander grading, a reduction per
    grading, and assembly into the Poincare count.
    """
    graph, assemble_key = whole_knot_graph(grid, backend=backend)
    return _run_to_result(graph, assemble_key,
                          mem_cap_bytes=mem_cap_bytes, vram_cap_bytes=vram_cap_bytes)


def _divide_by_V_once(p: dict) -> dict:
    """Divide a bigraded count by (1 + q^{-1} t^{-1}); solve from the top corner down."""
    if not p:
        return {}
    maslov_range = range(min(m for m, _ in p), max(m for m, _ in p) + 1)
    alex_range = range(min(a for _, a in p), max(a for _, a in p) + 1)
    quotient: dict = {}
    cells = [(m, a) for m in maslov_range for a in alex_range]
    for cell in sorted(cells, key=lambda c: c[0] + c[1], reverse=True):
        value = p.get(cell, 0) - quotient.get((cell[0] + 1, cell[1] + 1), 0)
        if value:
            quotient[cell] = value
    return quotient


def _tensor_V(h: dict, power: int) -> dict:
    p = dict(h)
    for _ in range(power):
        nxt: dict = defaultdict(int)
        for (m, a), c in p.items():
            nxt[(m, a)] += c
            nxt[(m - 1, a - 1)] += c
        p = {key: value for key, value in nxt.items() if value}
    return p


def hfk_hat(grid, *, backend: str = "bitint",
            mem_cap_bytes: int | None = None, vram_cap_bytes: int | None = None) -> dict:
    """HFK-hat as ``{(Maslov, Alexander): rank}``.

    Divides the grid Poincare polynomial by (1 + q^{-1} t^{-1})^{n-1} and verifies the
    quotient by reconstruction. With the grid in the standard chirality this matches
    KnotInfo directly.
    """
    grid_homology = grid_poincare(grid, backend=backend,
                                  mem_cap_bytes=mem_cap_bytes, vram_cap_bytes=vram_cap_bytes)
    quotient = grid_homology
    for _ in range(grid.n - 1):
        quotient = _divide_by_V_once(quotient)
    quotient = {key: value for key, value in quotient.items() if value}
    if any(value < 0 for value in quotient.values()) or _tensor_V(quotient, grid.n - 1) != grid_homology:
        raise UnvalidatedResult(
            "grid homology did not factor as HFK-hat (x) V^{n-1}; the V-factor division failed."
        )
    return quotient


def seifert_genus(grid, *, backend: str = "bitint",
                  mem_cap_bytes: int | None = None, vram_cap_bytes: int | None = None) -> int:
    """Seifert genus: the top Alexander grading carrying nonzero HFK-hat (genus detection)."""
    return max(a for _, a in hfk_hat(grid, backend=backend,
                                     mem_cap_bytes=mem_cap_bytes, vram_cap_bytes=vram_cap_bytes))
