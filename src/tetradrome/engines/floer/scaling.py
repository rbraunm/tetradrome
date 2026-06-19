# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Synthetic scaling grid and the reduction-memory projection (engine Phase 6).

``staircase_grid`` builds a valid n x n diagram for any n (the unknot on an n-grid), with the
full n! generators, so a scaling sweep can push the generator count past the small tabulated
knots and isolate the generation curve. ``dense_reduction_bytes`` turns a grading histogram
into the worst-case co-resident reduction footprint, using the shared per-block cost in
``algebra.memory``. Representative reduction cost needs real knot grids.
"""
from __future__ import annotations

from collections import defaultdict

from ...algebra import grading_peak_bytes
from .grid import GridDiagram


def staircase_grid(n: int) -> GridDiagram:
    """A valid n x n grid (O on the diagonal, X one step right) -- the unknot on an n-grid.
    Has the full n! generators, so it isolates generation scaling without a KnotInfo lookup."""
    if n < 2:
        raise ValueError("grid size must be at least 2.")
    return GridDiagram(list(range(n)), [(i + 1) % n for i in range(n)])


def dense_reduction_bytes(histogram: dict) -> int:
    """Worst-case co-resident dense F2 reduction memory (bytes) implied by a grading histogram.

    The packed reducer stores each degree-d boundary matrix of Alexander grading a as
    ``dims[a][d]`` bitmask columns of ``ceil(dims[a][d+1] / 64)`` uint64 words, plus the pivot
    columns it accumulates -- so its size is set by the DIMENSIONS, not the (sparse)
    differential (the shared per-block cost is ``algebra.memory.dense_block_bytes``). A worker
    reduces one grading at a time, degree by degree, so its peak is the largest such matrix in
    its grading; with more workers than Alexander gradings every grading can be co-resident, so
    the worst case is the sum over gradings of those per-grading peaks. This term scales as D^2
    and is what runs a large grid out of memory (validated: it is the bulk of the measured
    reduction footprint at n=10, and exceeds a 256 GiB box by n=11).
    """
    by_alexander: dict = defaultdict(dict)
    for (a_grading, degree), count in histogram.items():
        by_alexander[a_grading][degree] = count
    return sum(grading_peak_bytes(degrees) for degrees in by_alexander.values())
