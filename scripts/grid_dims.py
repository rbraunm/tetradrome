#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic memory model for the grid F2 reducer.

Histograms the generator count per (Alexander, degree=-Maslov) grading for a staircase grid,
WITHOUT building differentials or matrices -- so it is memory-safe at any n (it stores only
O(n^2) counts). From the dimensions it computes the EXACT dense reduction-matrix bytes the
reducer would allocate: the packed F2 reducer stores a degree-d matrix in Alexander grading a
as one bitmask column per dim(a,d) generator, each ceil(dim(a,d+1)/64) uint64 words --

    bytes(a, d) = dims[a][d] * ceil(dims[a][d+1] / 64) * 8

which depends on the DIMENSIONS only, not the differential. A worker reduces one grading at a
time, degree by degree, so its peak dense matrix is max over d of bytes(a, d). With far more
workers than Alexander gradings, every grading can be co-resident at once, so the worst-case
simultaneous dense footprint is the sum of those per-grading peaks. That is the number the OOM
limit is compared against -- computed here exactly, to predict fit/OOM at sizes we cannot run.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from multiprocessing import Pool

from tetradrome.engines.floer import GridDiagram, staircase_grid
from tetradrome.engines.floer.gradings import alexander, maslov
from tetradrome.engines.floer.scaling import _unrank


def _hist_slice(args):
    o_markers, x_markers, start, stop = args
    grid = GridDiagram(o_markers, x_markers)
    hist: dict = defaultdict(int)
    for k in range(start, stop):
        state = _unrank(k, grid.n)
        hist[(alexander(grid, state), -maslov(grid, state))] += 1
    return dict(hist)


def histogram(grid, workers: int) -> dict:
    """{(alexander, degree): count} over all n! generators, computed in parallel slices."""
    total = math.factorial(grid.n)
    if workers <= 1 or total < 2 * workers:
        return _hist_slice((grid.O, grid.X, 0, total))
    slices = [(grid.O, grid.X, total * w // workers, total * (w + 1) // workers)
              for w in range(workers)]
    merged: dict = defaultdict(int)
    with Pool(workers) as pool:
        for part in pool.imap_unordered(_hist_slice, slices, chunksize=1):
            for key, count in part.items():
                merged[key] += count
    return dict(merged)


def dense_model(hist: dict) -> dict:
    """From the (alexander, degree) histogram, the exact dense reduction-matrix bytes."""
    by_a: dict = defaultdict(dict)
    for (a, d), count in hist.items():
        by_a[a][d] = count

    per_grading_peak: dict = {}
    largest_matrix = 0
    for a, degs in by_a.items():
        peak = 0
        for d, ncols in degs.items():
            nrows = degs.get(d + 1, 0)
            nwords = max(1, (nrows + 63) // 64)
            mat = ncols * nwords * 8
            peak = max(peak, mat)
        per_grading_peak[a] = peak
        largest_matrix = max(largest_matrix, peak)

    return {
        "alex_gradings": len(by_a),
        "largest_matrix_bytes": largest_matrix,
        "coresident_bytes": sum(per_grading_peak.values()),   # all gradings at once
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-gib", type=float, default=200.0,
                        help="memory budget to compare the dense bound against")
    args = parser.parse_args()

    gib = 2 ** 30
    header = f"{'n':>3}{'n!':>15}{'alexGradings':>14}{'largestMat(GiB)':>17}{'denseCoresident(GiB)':>22}{'verdict':>10}"
    print(header)
    print("-" * len(header))
    for n in args.sizes:
        hist = histogram(staircase_grid(n), args.workers)
        model = dense_model(hist)
        coresident_gib = model["coresident_bytes"] / gib
        verdict = "OOM" if coresident_gib > args.limit_gib else "fits"
        print(f"{n:>3}{math.factorial(n):>15,}{model['alex_gradings']:>14}"
              f"{model['largest_matrix_bytes'] / gib:>17.2f}{coresident_gib:>22.2f}{verdict:>10}")


if __name__ == "__main__":
    main()
