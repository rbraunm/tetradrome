#!/usr/bin/env python3
"""Full grid-Floer validation sweep against KnotInfo -- portable, hardware-adaptive.

Runs the derived roster (floer_roster) one knot at a time, each using all available cores for
the HFK reduction; tau is cheap and serial; the genus comes from the same HFK ranks, so each
knot is computed once. Only one complex is ever live, so memory is bounded by the single
largest knot rather than by a concurrency choice -- there is nothing to hand-tune. Cores come
from the cpuset the process is actually allowed (sched_getaffinity), RAM from /proc/meminfo,
so the same invocation adapts from a laptop to labradorite; the only host-specific input is
--max-n. A knot whose predicted complex would not fit the detected RAM is skipped loudly
(never silently OOM'd) -- the decisions/0008 gate applied per box.

HFK and tau are compared up to mirror: "exact" matches KnotInfo directly, "mirror" matches the
mirror (a chirality-convention finding, D1, not a correctness failure), "MISMATCH" is real
disagreement. Genus is mirror-invariant. Exit status is nonzero if any knot mismatched or was
skipped for memory, so an unattended run signals trouble. See
roadmap/design/floer-phase-6-plan.md.

Usage (e.g. via the provisioner on labradorite):
    python3 scripts/sweep_floer.py --max-n 10
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.engines.floer import GridDiagram, floer_roster, hfk_hat, tau
from tetradrome.errors import BackendUnavailable


def core_count() -> int:
    try:
        return len(os.sched_getaffinity(0))   # respects container cpusets
    except AttributeError:                     # not Linux
        return os.cpu_count() or 1


def available_ram_bytes() -> int:
    """Best-effort available RAM: MemAvailable on Linux, else total physical, else 0."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


def _mirror_hfk(ranks: dict) -> dict:
    return {(-m, -a): r for (m, a), r in ranks.items()}


def _classify(name: str, ranks: dict, tau_value: int, genus: int) -> tuple[str, str, str]:
    ki_hfk = ki.hfk_ranks(name)
    if ranks == ki_hfk:
        hfk = "exact"
    elif ranks == _mirror_hfk(ki_hfk):
        hfk = "mirror"
    else:
        hfk = "MISMATCH"
    ki_tau = ki.tau_invariant(name)
    tau_s = "exact" if tau_value == ki_tau else ("mirror" if tau_value == -ki_tau else "MISMATCH")
    genus_s = "exact" if genus == int(ki.lookup(name)["three_genus"]) else "MISMATCH"
    return hfk, tau_s, genus_s


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate native grid Floer against KnotInfo.")
    ap.add_argument("--max-n", type=int, default=10,
                    help="largest grid number to include (default 10, the brute floor)")
    ap.add_argument("--min-n", type=int, default=0,
                    help="smallest grid number to include (for partial reruns)")
    ap.add_argument("--workers", type=int, default=0,
                    help="reduction workers per knot (default: all available cores)")
    ap.add_argument("--bytes-per-gen", type=float, default=6000.0,
                    help="bytes per generator for the memory projection; the default reflects "
                    "the reduction peak (anchored to the n=11 ~200 GiB overrun), deliberately "
                    "above the bench generation figure so the gate fails safe -- override per box")
    ap.add_argument("--mem-fraction", type=float, default=0.8,
                    help="fraction of detected RAM treated as the budget")
    args = ap.parse_args()

    try:
        roster = [(nm, n) for nm, n in floer_roster(args.max_n) if n >= args.min_n]
    except BackendUnavailable as exc:
        print(f"KnotInfo backend unavailable: {exc}", file=sys.stderr)
        return 2

    cores = core_count()
    workers = args.workers or cores
    ram = available_ram_bytes()
    budget = int(ram * args.mem_fraction) if ram else 0

    def predicted(n: int) -> int:
        return int(math.factorial(n) * args.bytes_per_gen)

    gib = lambda b: f"{b / 2**30:.0f} GiB"
    print(f"sweep: {len(roster)} knots (n {args.min_n}..{args.max_n}); "
          f"cores={cores} workers/knot={workers}; "
          f"RAM={'unknown' if not ram else gib(ram)} budget={'n/a' if not budget else gib(budget)}",
          flush=True)

    n_exact = n_mirror = n_fail = n_skip = 0
    t_all = time.perf_counter()
    for name, n in roster:
        if budget and predicted(n) > budget:
            print(f"  SKIP     {name:10} n={n:<2} predicted {predicted(n)/2**30:.1f} GiB "
                  f"> {gib(budget)} budget", flush=True)
            n_skip += 1
            continue
        t0 = time.perf_counter()
        try:
            grid = GridDiagram.from_knotinfo(name)
            ranks = hfk_hat(grid, workers=workers)
            genus = max(a for _m, a in ranks)
            tau_value = tau(grid)
            hfk_s, tau_s, genus_s = _classify(name, ranks, tau_value, genus)
        except Exception as exc:  # noqa: BLE001 -- report which knot failed, keep sweeping
            print(f"  ERROR    {name:10} n={n:<2} {type(exc).__name__}: {exc}", flush=True)
            n_fail += 1
            continue
        dt = time.perf_counter() - t0
        if "MISMATCH" in (hfk_s, tau_s, genus_s):
            worst, n_fail = "MISMATCH", n_fail + 1
        elif "mirror" in (hfk_s, tau_s):
            worst, n_mirror = "mirror", n_mirror + 1
        else:
            worst, n_exact = "exact", n_exact + 1
        print(f"  {worst:8} {name:10} n={n:<2} hfk={hfk_s} tau={tau_s} genus={genus_s} "
              f"({dt:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t_all
    print(f"\ndone in {elapsed:.0f}s: {n_exact} exact, {n_mirror} mirror, "
          f"{n_fail} mismatch/error, {n_skip} skipped (memory).", flush=True)
    return 0 if (n_fail == 0 and n_skip == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
