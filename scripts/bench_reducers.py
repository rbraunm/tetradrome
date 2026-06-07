#!/usr/bin/env python3
"""Compare the F2 reduction tiers on accuracy and speed.

Accuracy is the gate: every available tier must reproduce the reference homology on the
real Khovanov catalog, or the run reports FAIL. Speed is then measured two ways -- on the
real knot complexes, and on synthetic dense F2 matrices across a size sweep (where any
GPU advantage, if present, shows up). Backends that are not installed (e.g. packed-gpu
with no CUDA device) are skipped with a note rather than failing the run.

Usage (with the package installed, e.g. `pip install -e .`):
    python3 scripts/bench_reducers.py
    python3 scripts/bench_reducers.py --sizes 512,1024,2048 --knots 3_1,8_19
    python3 scripts/bench_reducers.py --backends bitint,packed-cpu,packed-gpu
    python3 scripts/bench_reducers.py --skip-synthetic

On a machine with a CUDA GPU and cupy installed, `packed-gpu` joins automatically. Note:
GPU wins are expected only in the large dense regime; the per-column reducer is
Python-loop-bound, and small knot complexes generally favour the CPU tiers. Measure, do
not assume.
"""
from __future__ import annotations

import argparse
import os
import random
import time

from tetradrome import knots
from tetradrome.algebra import gpu, memory, tiers
from tetradrome.algebra.parallel import parallel_f2_homology
from tetradrome.algebra.reduce_reference import homology
from tetradrome.engines import khovanov

DEFAULT_KNOTS = ["3_1", "4_1", "5_2", "6_2", "7_4"]
DEFAULT_SIZES = [128, 256, 512]


def _selected_backends(requested: list[str] | None) -> list[tuple[str, str]]:
    avail = tiers.available_f2_backends()
    note = {name: n for name, _, n in avail}
    ok = {name: a for name, a, _ in avail}
    names = requested if requested else [n for n, _, _ in avail]
    out = []
    for name in names:
        if name not in ok:
            print(f"  ! unknown backend {name!r}, skipping")
        elif not ok[name]:
            print(f"  - {name:11s} unavailable ({note[name]}), skipping")
        else:
            out.append((name, note[name]))
    return out


def _print_gpu() -> None:
    print(gpu.format_report())
    steps = gpu.enablement_instructions()
    if steps:
        print("\nTo enable the GPU tier:")
        for line in steps.splitlines():
            print(f"  {line}")
    print()


def _print_availability() -> None:
    print("F2 reduction backends:")
    for name, ok, note in tiers.available_f2_backends():
        mark = "available" if ok else "unavailable"
        print(f"  {name:11s} {mark:12s} {note}")
    print()


def accuracy(knot_names: list[str], backends: list[str]) -> None:
    print("== accuracy (every tier must equal the reference homology) ==")
    complexes = []
    for name in knot_names:
        pd = knots.from_name(name).pd_code
        for j, cx in khovanov.khovanov_complexes(pd).items():
            complexes.append((name, j, cx, homology(cx, verify=False)))
    for backend in backends:
        rank_fn_ok = True
        mismatches = []
        for name, j, cx, ref in complexes:
            got = tiers.f2_homology(cx, backend=backend)
            if got != ref:
                mismatches.append(f"{name} q={j}: {got} != {ref}")
        status = "PASS" if not mismatches else f"FAIL ({len(mismatches)})"
        print(f"  {backend:11s} {status}")
        for m in mismatches[:5]:
            print(f"      {m}")
    print()


def _time(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def speed_knots(knot_names: list[str], backends: list[str], repeat: int) -> None:
    print(f"== speed: knot Khovanov homology (best of {repeat}) ==")
    cxs = []
    for name in knot_names:
        pd = knots.from_name(name).pd_code
        cxs.extend(khovanov.khovanov_complexes(pd).values())
    print(f"  {len(knot_names)} knots, {len(cxs)} quantum complexes")
    for backend in backends:
        secs = _time(lambda: [tiers.f2_homology(cx, backend=backend) for cx in cxs], repeat)
        print(f"  {backend:11s} {secs * 1e3:9.2f} ms")
    print()


def _random_f2_columns(n: int, density: float, seed: int) -> list[set[int]]:
    rng = random.Random(seed)
    return [{r for r in range(n) if rng.random() < density} for _ in range(n)]


def speed_synthetic(sizes: list[int], backends: list[str], density: float, repeat: int) -> None:
    print(f"== speed: synthetic dense F2 rank, density={density} (best of {repeat}) ==")
    header = "  size    " + "".join(f"{b:>14s}" for b in backends)
    print(header)
    for n in sizes:
        cols = _random_f2_columns(n, density, seed=n)
        cells = []
        for backend in backends:
            fn = tiers.rank_backend(backend)
            secs = _time(lambda: fn(cols, n), repeat)
            cells.append(f"{secs * 1e3:11.2f}ms")
        print(f"  {n:<7d} " + "".join(f"{c:>14s}" for c in cells))
    print()


_CPU_BACKENDS = ("reference", "bitint", "packed-cpu")


def speed_parallel(knot_names: list[str], backends: list[str], workers: int, repeat: int) -> None:
    cpu_backends = [b for b in backends if b in _CPU_BACKENDS]
    if not cpu_backends:
        return
    items = {}
    for name in knot_names:
        pd = knots.from_name(name).pd_code
        for j, cx in khovanov.khovanov_complexes(pd).items():
            items[(name, j)] = cx
    print(f"== speed: serial vs parallel, {len(items)} complexes on {workers} workers "
          f"(best of {repeat}) ==")
    print(f"  {'backend':11s} {'serial':>12s} {'parallel':>12s} {'speedup':>10s}")
    for backend in cpu_backends:
        ser = _time(lambda: {k: tiers.f2_homology(cx, backend=backend) for k, cx in items.items()}, repeat)
        par = _time(lambda: parallel_f2_homology(items, backend=backend, workers=workers), repeat)
        speedup = ser / par if par > 0 else float("inf")
        print(f"  {backend:11s} {ser * 1e3:10.2f}ms {par * 1e3:10.2f}ms {speedup:9.2f}x")
    print()


def routing(knot_names: list[str]) -> None:
    cfg = gpu.gpu_config()
    available = tiers.available_f2_backends()
    print("== memory prediction & routing (heaviest quantum complex per knot) ==")
    print(f"  {'knot':6s} {'peak (packed)':>15s} {'route':>12s}   reason")
    for name in knot_names:
        pd = knots.from_name(name).pd_code
        worst = None
        for cx in khovanov.khovanov_complexes(pd).values():
            d = memory.route_backend(cx, available, gpu_cfg=cfg)
            if worst is None or d.predicted_bytes > worst.predicted_bytes:
                worst = d
        kib = worst.predicted_bytes / 1024
        print(f"  {name:6s} {kib:12.1f} KiB {worst.backend:>12s}   {worst.reason}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knots", type=str, default=",".join(DEFAULT_KNOTS),
                    help="comma-separated knot names")
    ap.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)),
                    help="comma-separated synthetic matrix sizes")
    ap.add_argument("--backends", type=str, default=None,
                    help="comma-separated backends (default: all available)")
    ap.add_argument("--density", type=float, default=0.5, help="synthetic matrix density")
    ap.add_argument("--repeat", type=int, default=3, help="timing repetitions (best is kept)")
    ap.add_argument("--skip-synthetic", action="store_true", help="skip the synthetic sweep")
    ap.add_argument("--skip-parallel", action="store_true", help="skip the serial-vs-parallel run")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes for the parallel run (default: CPU count)")
    ap.add_argument("--gpu-info", action="store_true",
                    help="print GPU detection and enablement guidance, then exit")
    args = ap.parse_args()

    _print_gpu()
    if args.gpu_info:
        return

    _print_availability()
    requested = args.backends.split(",") if args.backends else None
    backends = [name for name, _ in _selected_backends(requested)]
    if not backends:
        print("No runnable backends selected.")
        return
    print(f"Running backends: {', '.join(backends)}\n")

    knot_names = [k for k in args.knots.split(",") if k]
    accuracy(knot_names, backends)
    routing(knot_names)
    speed_knots(knot_names, backends, args.repeat)
    if not args.skip_parallel:
        speed_parallel(knot_names, backends, args.workers or (os.cpu_count() or 1), args.repeat)
    if not args.skip_synthetic:
        sizes = [int(s) for s in args.sizes.split(",") if s]
        speed_synthetic(sizes, backends, args.density, args.repeat)


if __name__ == "__main__":
    main()
