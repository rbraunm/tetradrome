#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic memory model for the grid F2 reducer.

Histograms generators per (Alexander, degree) grading for a staircase grid WITHOUT building
differentials or matrices (memory-safe at any n), then reports the exact dense reduction-matrix
bytes the packed reducer would allocate. The histogram comes from the engine
(``grading_histogram``); the cost functions live in the algebra layer (``max_grading_bytes``,
``dense_reduction_bytes``), and the same model backs ``bench_grid_floer.py``'s
``--mem-budget-gib`` guard.

Two figures: ``wavesFloor`` is the largest single grading's reduction peak -- the minimum
budget at which the workload is reducible at all (the bounded scheduler runs gradings in
deterministic waves, so a size fits when its largest grading fits). ``denseCoresident`` is the
sum over gradings, the memory only an unbounded number of co-resident workers would need. The
dense matrices scale as D^2, negligible at small n and dominant by n=10, so the projection
predicts the fit/OOM boundary at sizes too large to run.
"""
from __future__ import annotations

import argparse
import math

from tetradrome.algebra import dense_reduction_bytes, max_grading_bytes
from tetradrome.engines.floer import (
    grading_histogram,
    staircase_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-gib", type=float, default=200.0,
                        help="memory budget; a size fits when its largest single grading fits "
                             "(the bounded scheduler runs the rest in waves), OOM otherwise")
    args = parser.parse_args()

    gib = 2 ** 30
    header = (f"{'n':>3}{'n!':>15}{'alexGradings':>14}{'wavesFloor(GiB)':>17}"
              f"{'denseCoresident(GiB)':>22}{'verdict':>10}")
    print(header)
    print("-" * len(header))
    for n in args.sizes:
        histogram = grading_histogram(staircase_grid(n), args.workers)
        gradings = len({a for a, _ in histogram})
        floor_gib = max_grading_bytes(histogram) / gib
        coresident_gib = dense_reduction_bytes(histogram) / gib
        verdict = "fits" if floor_gib <= args.limit_gib else "OOM"
        print(f"{n:>3}{math.factorial(n):>15,}{gradings:>14}"
              f"{floor_gib:>17.2f}{coresident_gib:>22.2f}{verdict:>10}")


if __name__ == "__main__":
    main()
