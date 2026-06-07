#!/usr/bin/env python3
"""Scaling study for the grid (knot Floer) engine.

Measures, per grid, the two costs that the n! generator count splits into -- *generation*
(enumerating permutations and computing gradings + differentials) and *reduction* (the F2
linear algebra) -- plus peak traced memory, so the wall (time vs memory, generation vs
reduction) is located empirically rather than guessed. Run it on real hardware and read the
curve.

Examples:
    # tabulated knots, increasing grid size, default bitint reducer, serial
    python3 scripts/bench_grid_floer.py --knots 3_1 4_1 5_1 5_2 8_19

    # synthetic grids to push the generator count, with parallel generation + reduction
    python3 scripts/bench_grid_floer.py --sizes 5 6 7 8 9 --gen-workers 16 --workers 16

    # compare reducer backends on one knot
    python3 scripts/bench_grid_floer.py --knots 8_19 --backend reference
    python3 scripts/bench_grid_floer.py --knots 8_19 --backend packed-gpu
"""
from __future__ import annotations

import argparse
import math
import time
import tracemalloc

from tetradrome.engines.floer import (
    GridDiagram,
    grid_complexes,
    parallel_grid_complexes,
    reduce_complexes,
    staircase_grid,
)


def _time(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def measure(grid, *, backend, workers, pin, gen_workers):
    """Return (generation_s, reduction_s, peak_MiB, generator_count, support_size)."""
    tracemalloc.start()
    if gen_workers > 1:
        complexes, gen_s = _time(lambda: parallel_grid_complexes(grid, gen_workers))
    else:
        complexes, gen_s = _time(lambda: grid_complexes(grid))
    poincare, red_s = _time(
        lambda: reduce_complexes(complexes, backend=backend, workers=workers, pin=pin)
    )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return gen_s, red_s, peak / 2**20, math.factorial(grid.n), len(poincare)


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
    args = parser.parse_args()

    targets = ([(name, GridDiagram.from_knotinfo(name)) for name in args.knots] +
               [(f"staircase-{n}", staircase_grid(n)) for n in args.sizes])
    if not targets:
        parser.error("give --knots and/or --sizes")

    print(f"backend={args.backend}  reduction-workers={args.workers}  "
          f"gen-workers={args.gen_workers}  pin={args.pin}\n")
    header = f"{'target':<14}{'n':>3}{'n!':>12}{'gen(s)':>10}{'reduce(s)':>11}{'peak(MiB)':>11}{'(M,A)':>7}"
    print(header)
    print("-" * len(header))
    for name, grid in targets:
        gen_s, red_s, peak, count, support = measure(
            grid, backend=args.backend, workers=args.workers, pin=args.pin,
            gen_workers=args.gen_workers,
        )
        print(f"{name:<14}{grid.n:>3}{count:>12,}{gen_s:>10.3f}{red_s:>11.3f}"
              f"{peak:>11.1f}{support:>7}")


if __name__ == "__main__":
    main()
