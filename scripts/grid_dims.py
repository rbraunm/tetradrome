#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic memory model for the grid F2 reducer.

Histograms generators per (Alexander, degree) grading for a staircase grid WITHOUT building
differentials or matrices (memory-safe at any n), then reports the exact dense reduction-matrix
bytes the packed reducer would allocate. The histogram comes from the engine
(``grading_histogram``); the cost functions live in the algebra layer (``max_grading_bytes``,
``dense_reduction_bytes``), and the same model is what the scheduler prices the Floer reduction
graph against.

Two figures: ``largestGrad`` is the largest single grading's reduction peak -- the minimum
budget at which the workload is reducible at all, since the scheduler holds the live peak below
the budget by ordering reductions and spilling held results, so a size is feasible when its
largest grading fits. ``denseCoresident`` is the sum over gradings, the memory unbounded
concurrency (every grading reduced at once) would need. The dense matrices scale as D^2,
negligible at small n and dominant by n=10, so the projection predicts the fit/OOM boundary at
sizes too large to run.
"""
from __future__ import annotations

import argparse
import math

from tetradrome.algebra import dense_reduction_bytes, max_grading_bytes
from tetradrome.engines.floer import grading_histogram, staircase_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--limit-gib", type=float, default=200.0,
                        help="memory budget; a size fits when its largest single grading fits "
                             "(the scheduler orders and spills the rest), OOM otherwise")
    args = parser.parse_args()

    gib = 2 ** 30
    header = (f"{'n':>3}{'n!':>15}{'alexGradings':>14}{'largestGrad(GiB)':>18}"
              f"{'denseCoresident(GiB)':>22}{'verdict':>10}")
    print(header)
    print("-" * len(header))
    for n in args.sizes:
        histogram = grading_histogram(staircase_grid(n))
        gradings = len({a for a, _ in histogram})
        floor_gib = max_grading_bytes(histogram) / gib
        coresident_gib = dense_reduction_bytes(histogram) / gib
        verdict = "fits" if floor_gib <= args.limit_gib else "OOM"
        print(f"{n:>3}{math.factorial(n):>15,}{gradings:>14}"
              f"{floor_gib:>18.2f}{coresident_gib:>22.2f}{verdict:>10}")


if __name__ == "__main__":
    main()
