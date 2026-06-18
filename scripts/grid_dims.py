#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic memory model for the grid F2 reducer.

Histograms generators per (Alexander, degree) grading for a staircase grid WITHOUT building
differentials or matrices (memory-safe at any n), then reports the exact dense reduction-matrix
bytes the packed reducer would allocate. The model and its inputs live in the engine
(``grading_histogram`` / ``dense_reduction_bytes``); this is the reporting front end, and the
same projection backs ``bench_grid_floer.py``'s ``--mem-budget-gib`` guard.

The dense matrices scale as D^2, so they are negligible at small n, dominate the footprint by
n=10, and exceed a 256 GiB box by n=11 -- the projection predicts that fit/OOM boundary at
sizes too large to run.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

from tetradrome.engines.floer import (
    dense_reduction_bytes,
    grading_histogram,
    staircase_grid,
)


def _report_fields(histogram: dict) -> tuple[int, int]:
    """(#Alexander gradings, largest single dense matrix in bytes) for the detailed table."""
    by_alexander: dict = defaultdict(dict)
    for (a_grading, degree), count in histogram.items():
        by_alexander[a_grading][degree] = count
    largest = 0
    for degrees in by_alexander.values():
        for degree, ncols in degrees.items():
            nwords = max(1, (degrees.get(degree + 1, 0) + 63) // 64)
            largest = max(largest, ncols * nwords * 8)
    return len(by_alexander), largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-gib", type=float, default=200.0,
                        help="memory budget to compare the dense bound against")
    args = parser.parse_args()

    gib = 2 ** 30
    header = (f"{'n':>3}{'n!':>15}{'alexGradings':>14}{'largestMat(GiB)':>17}"
              f"{'denseCoresident(GiB)':>22}{'verdict':>10}")
    print(header)
    print("-" * len(header))
    for n in args.sizes:
        histogram = grading_histogram(staircase_grid(n), args.workers)
        gradings, largest = _report_fields(histogram)
        coresident_gib = dense_reduction_bytes(histogram) / gib
        verdict = "OOM" if coresident_gib > args.limit_gib else "fits"
        print(f"{n:>3}{math.factorial(n):>15,}{gradings:>14}"
              f"{largest / gib:>17.2f}{coresident_gib:>22.2f}{verdict:>10}")


if __name__ == "__main__":
    main()
