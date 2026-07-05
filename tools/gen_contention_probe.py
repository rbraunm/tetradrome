#!/usr/bin/env python3
"""Generation contention probe for the grid-Floer engine (diagnostic).

Question this answers: is grid-Floer generation compute-bound or
memory/allocation-bound? That decides the lever (JIT/vectorize vs a packed
rewrite that kills per-state object churn) and whether more cores ever help.

Method: run the REAL per-state work (_generate_slice, alloc-heavy) on N cores
at once -- pinned, all released together by a barrier -- across a
topology-aware sweep (solo / one NUMA node / all physical / all logical), and
watch per-state time degrade with N. Run an allocation-LIGHT twin
(_grading_slice: gradings only, no per-state tuples/sets) through the same
sweep. The gap between the two degradation curves separates allocation
contention from arithmetic/bandwidth contention; process_time vs wall tells
stalls (more cycles per state) from descheduling.

Reading: if gen-x climbs while grad-x stays flat through the physical-core
points (1 -> one node -> all phys), the cost is per-state allocation -> a
packed rewrite is the win and more cores will not help. If both climb together
across those points, it is arithmetic/bandwidth on the work itself -> JIT or
vectorize helps but watch bandwidth. A jump only at the all-logical point is
just HT oversubscription -> physical cores still scale.

Uses only existing tetradrome functions. Peak RAM ~= N * SLICE * ~3KB
(<= ~20 GB at N=88), well inside 200 GB.
"""
import os
import time
import resource
import statistics
import multiprocessing as mp

from tetradrome.engines.floer.grid import staircase_grid
from tetradrome.engines.floer.generation import _generate_slice, _grading_slice

N_GRID = 10
SLICE = 40000            # states per worker; 88 * 40000 < 10! and within RAM
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
    """(node_cpus, node_of, ordered, reps_first_node, n_phys, n_log, fallback).

    ordered lists allowed cpus as: node0 physical reps, node1 physical reps,
    ..., then the 2nd HT sibling of each -- so ordered[:k] grows by locality
    (fills one socket's physical cores, then the other, then HT siblings)."""
    try:
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
            raise RuntimeError("empty topology")

        by_node = {}
        for rep, sibs in phys:
            by_node.setdefault(node_of.get(rep, 0), []).append(sibs)
        ordered = []
        max_depth = max(len(s) for _, s in phys)
        for depth in range(max_depth):
            for nid in sorted(by_node):
                for sibs in by_node[nid]:
                    if depth < len(sibs):
                        ordered.append(sibs[depth])

        first_set = set(node_cpus[sorted(node_cpus)[0]])
        reps_first_node = sum(1 for rep, _ in phys if rep in first_set)
        return node_cpus, node_of, ordered, reps_first_node, len(phys), len(ordered), False
    except Exception as exc:
        print(f"# topology read failed ({exc}); naive fallback "
              f"(NUMA/HT labels unreliable)", flush=True)
        a = sorted(allowed)
        return {0: a}, {c: 0 for c in a}, a, len(a), len(a), len(a), True


def worker(o_markers, x_markers, kind, core, barrier, queue):
    try:
        os.sched_setaffinity(0, {core})
        barrier.wait(timeout=300)            # release all workers together -> real contention
        wall0, cpu0 = time.perf_counter(), time.process_time()
        if kind == "gen":
            result = _generate_slice((o_markers, x_markers, 0, SLICE))
            produced = len(result)
        else:
            result = _grading_slice((o_markers, x_markers, 0, SLICE))
            produced = sum(result.values())
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
     n_phys, n_log, fallback) = read_topology(allowed)
    sweep = sorted({1, reps_first_node, n_phys, n_log})
    sweep = [n for n in sweep if n <= len(allowed)]

    grid = staircase_grid(N_GRID)
    o_markers, x_markers = grid.O, grid.X

    print(f"# generation contention probe  grid n={N_GRID}  slice={SLICE} states/worker")
    print("# allowed cores: %d   NUMA: %s" % (
        len(allowed),
        "; ".join(f"node{nid}={len(c)}cpu" for nid, c in sorted(node_cpus.items()))))
    print(f"# physical cores: {n_phys}   logical: {n_log}   fallback={fallback}")
    print(f"# sweep N: {sweep}  (1=solo, {reps_first_node}=one NUMA node phys, "
          f"{n_phys}=all phys, {n_log}=all logical)")
    print("# x = per-state wall vs the N=1 baseline of the SAME work-kind; "
          "compare gen-x against grad-x\n")

    for kind, label in (("gen", "generate(alloc-heavy)"), ("grad", "gradings(alloc-light)")):
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
