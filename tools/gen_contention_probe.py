#!/usr/bin/env python3
"""NUMA contention probe for the grid-Floer kernels (diagnostic).

Question: do the scheduled grid kernels keep their per-state speed as more cores run them at
once -- on one NUMA node's physical cores, on both nodes' physical cores, and on every logical
CPU? The answer decides whether the scheduler should fill both nodes with one calculation's jobs
or give each node its own calculation, and whether hyperthread siblings are worth using.

Method: run a real slice job on N cores at once, each worker pinned to one CPU and all released
together by a barrier, for N = 1, one node's physical cores, all physical cores, all logical
CPUs. Two kinds of job, as the scheduler runs them:
  generation  -- gradings, differential and target ranks for a slice (`_generate_slice`)
  histogram   -- gradings only (`_histogram_slice`)
The kernels are bound in the parent before the workers fork (`prewarm`), so no worker compiles
inside its timed region.

Reading: x is per-state wall time against the same kind's N=1 baseline. Flat through all physical
cores: the work is compute-bound and both nodes scale, so one calculation may spread over both.
Climbing at one node's physical cores: shared bandwidth or cache inside a node. Worse on the
second node than the first (the per-node line): placement or remote memory. A jump only at all
logical CPUs: hyperthread siblings share a core and add little. cpu/st near wall/st means the
workers ran, not waited.

Peak RAM is about N * SLICE * 2 KB at grid number 10.
"""
import os
import time
import resource
import statistics
import multiprocessing as mp

from tetradrome.engines.floer import prewarm  # noqa: F401 -- binds the kernels before forking
from tetradrome.engines.floer.grid import staircase_grid
from tetradrome.engines.floer.generation import _generate_slice, _histogram_slice

N_GRID = 10
SLICE = 40000            # states per worker; every worker takes the same first slice
REPEAT_BASELINE = 3


def parse_cpulist(text):
    out = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def read_topology(allowed):
    """(node_cpus, node_of, ordered, reps_first_node, n_phys, n_log).

    ordered lists allowed cpus as: node0 physical reps, node1 physical reps,
    ..., then the 2nd HT sibling of each -- so ordered[:k] grows by locality
    (fills one socket's physical cores, then the other, then HT siblings)."""
    node_cpus = {}
    for entry in sorted(os.listdir("/sys/devices/system/node")):
        if not entry.startswith("node"):
            continue
        with open(f"/sys/devices/system/node/{entry}/cpulist") as fh:
            cpus = sorted(c for c in parse_cpulist(fh.read()) if c in allowed)
        if cpus:
            node_cpus[int(entry[4:])] = cpus
    node_of = {c: nid for nid, cpus in node_cpus.items() for c in cpus}

    seen = set()
    phys = []          # (representative_cpu, [sibling cpus])
    for cpu in sorted(allowed):
        if cpu in seen:
            continue
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as fh:
            sibs = sorted(c for c in parse_cpulist(fh.read()) if c in allowed)
        seen.update(sibs)
        phys.append((min(sibs), sibs))
    if not node_cpus or not phys:
        raise RuntimeError("no NUMA or core topology under /sys for the allowed CPUs")

    by_node = {}
    for rep, sibs in phys:
        by_node.setdefault(node_of[rep], []).append(sibs)
    ordered = []
    max_depth = max(len(s) for _, s in phys)
    for depth in range(max_depth):
        for nid in sorted(by_node):
            for sibs in by_node[nid]:
                if depth < len(sibs):
                    ordered.append(sibs[depth])

    first_set = set(node_cpus[sorted(node_cpus)[0]])
    reps_first_node = sum(1 for rep, _ in phys if rep in first_set)
    return node_cpus, node_of, ordered, reps_first_node, len(phys), len(ordered)


def worker(o_markers, x_markers, kind, core, barrier, queue):
    try:
        os.sched_setaffinity(0, {core})
        barrier.wait(timeout=300)            # release all workers together -> real contention
        wall0, cpu0 = time.perf_counter(), time.process_time()
        if kind == "gen":
            result = _generate_slice((o_markers, x_markers, 0, SLICE))
            produced = len(result.maslov)
        else:
            result = _histogram_slice((o_markers, x_markers, 0, SLICE))
            produced = sum(result.values())
        if produced != SLICE:
            raise RuntimeError(f"{kind} slice covered {produced} states, expected {SLICE}")
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        del result
        queue.put((core, wall, cpu, rss_kb, produced))
    except Exception as exc:
        queue.put(("ERR", core, repr(exc)))


def run_batch(o_markers, x_markers, kind, cores):
    ctx = mp.get_context("fork")             # inherit warm imports; timed region is pure compute
    barrier = ctx.Barrier(len(cores))
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(o_markers, x_markers, kind, c, barrier, queue))
             for c in cores]
    for p in procs:
        p.start()
    rows = []
    for _ in cores:
        item = queue.get(timeout=900)
        if item[0] == "ERR":
            for p in procs:
                p.terminate()
            raise RuntimeError(f"worker on core {item[1]} failed: {item[2]}")
        rows.append(item)
    for p in procs:
        p.join()
    return rows


def summarize(label, rows, baseline_wall_us, node_of):
    wall_us = sorted(w / SLICE * 1e6 for _, w, _, _, _ in rows)
    cpu_us = sorted(c / SLICE * 1e6 for _, _, c, _, _ in rows)
    rss_mb = sorted(r / 1024.0 for _, _, _, r, _ in rows)
    med_w = statistics.median(wall_us)
    mult = med_w / baseline_wall_us if baseline_wall_us else 1.0
    print(f"{label:>24} | N={len(rows):>3} | wall/st {med_w:7.2f}us "
          f"(min {wall_us[0]:6.2f} max {wall_us[-1]:6.2f}) | "
          f"cpu/st {statistics.median(cpu_us):7.2f}us | "
          f"rss {statistics.median(rss_mb):6.0f}MB | x{mult:5.2f}")
    nodes_present = {node_of.get(c) for c, *_ in rows}
    if len(nodes_present) > 1:
        per_node = {}
        for core, w, *_ in rows:
            per_node.setdefault(node_of.get(core), []).append(w / SLICE * 1e6)
        parts = "  ".join(f"node{nid}:{statistics.median(v):.2f}us(n={len(v)})"
                          for nid, v in sorted(per_node.items()))
        print(f"{'':>24} |     per-node wall/st: {parts}")
    return med_w


def main():
    allowed = set(os.sched_getaffinity(0))
    (node_cpus, node_of, ordered, reps_first_node,
     n_phys, n_log) = read_topology(allowed)
    sweep = sorted({1, reps_first_node, n_phys, n_log})
    sweep = [n for n in sweep if n <= len(allowed)]

    grid = staircase_grid(N_GRID)
    o_markers, x_markers = grid.O, grid.X

    print(f"# NUMA contention probe  grid n={N_GRID}  slice={SLICE} states/worker")
    print("# allowed cores: %d   NUMA: %s" % (
        len(allowed),
        "; ".join(f"node{nid}={len(c)}cpu" for nid, c in sorted(node_cpus.items()))))
    print(f"# physical cores: {n_phys}   logical: {n_log}")
    print(f"# sweep N: {sweep}  (1=solo, {reps_first_node}=one NUMA node phys, "
          f"{n_phys}=all phys, {n_log}=all logical)")
    print("# x = per-state wall vs the N=1 baseline of the same kind of job\n")

    for kind, label in (("gen", "generation slice"), ("hist", "histogram slice")):
        base_core = node_cpus[sorted(node_cpus)[0]][0]
        base_rows = []
        for _ in range(REPEAT_BASELINE):
            base_rows += run_batch(o_markers, x_markers, kind, [base_core])
        baseline = summarize(f"{label} baseline", base_rows, None, node_of)
        for n in sweep:
            if n == 1:
                continue
            run = run_batch(o_markers, x_markers, kind, ordered[:n])
            summarize(label, run, baseline, node_of)
        print()


if __name__ == "__main__":
    main()
