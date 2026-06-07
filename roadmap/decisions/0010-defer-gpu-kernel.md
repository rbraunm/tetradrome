# 0010 - Defer the on-device GPU F2 rank kernel

**Status:** Accepted

## Context

Phase 5 added GPU support to the reduction back end. The first-cut `packed-gpu` tier
(`f2_rank_dense` — vectorized dense row reduction over an array module, run on cupy) is
*correct* on real hardware: it reproduces the reference homology across the whole catalog
on an RTX-class card (`sm_75`). But it is not competitive on speed. Measured on that card,
at a 2048-square dense F2 matrix it ran ~14x *slower* than the pure-Python `bitint`
reducer (≈6.8 s vs ≈0.47 s), and the gap widens with size.

The cause is structural, not a tuning detail:

- The reducer issues **one host↔device sync per column** — picking the pivot row for
  partial pivoting requires reading a value back to the host — so the cost is O(`ncols`)
  serial round-trips whose count grows with the matrix width.
- It operates on **unpacked `uint8`**, 8x the memory traffic of a bit-packed layout.
- The only genuinely parallel step (a vectorized row XOR) is swamped by those serial
  syncs.

The broader measurement across the entire acceleration phase is the important finding:
the **pure-Python, dependency-free `bitint` reducer is the workhorse** — it beats numpy,
roughly matches numba, and beats this first-cut GPU kernel in every tested regime, and the
knot complexes we actually compute are tiny and route to the CPU anyway.

A GPU win is plausible only at large scale and only after substantial engineering — a
genuinely on-device kernel. The value of that work is not merely "faster": sufficient
reduction throughput could open computational avenues that do not exist today simply
because nothing has yet pushed hard enough to need them — complexes (very high crossing
number, large Lee or Floer cubes) too big for the CPU reducers to finish in acceptable
time or memory. There is no such pressure yet.

## Decision

Defer a proper on-device GPU F2 rank kernel as a **late-project goal**, taken up only when
(a) higher-value work is done and (b) there is a concrete target computation the CPU tiers
cannot complete in acceptable time/memory. This is speed/scale tuning, not new capability,
and must not preempt capability work (e.g. the Floer engine).

- **Keep the current correct `packed-gpu` tier** in place behind the router. The memory
  gate (ADR 0008) already declines to select it below a size threshold, so it is never
  chosen in the range where it loses. It stays as (i) a validated correctness oracle the
  eventual kernel must agree with, and (ii) a fallback for any future very-large case.
- **When built, the kernel must clear the same agreement bar** (`== reference`) and should
  target the structural problems identified above: bit-packed `uint64` columns (cut memory
  and XOR traffic 8x), and an on-device pivoting/elimination scheme that removes the
  per-column host sync — e.g. a custom cupy `RawKernel` doing block-wise elimination with
  warp-level leading-bit detection, or an M4RI-style (Method of the Four Russians) GF(2)
  approach suited to SIMT. Its measured crossover against `bitint` then calibrates the
  router threshold.

## Consequences

- Phase 5 is otherwise complete: the acceleration tiers exist and are validated, and the
  operative empirical result — `bitint` is the workhorse, the system auto-routes to it —
  stands. No effort is spent trying to beat an already-excellent pure-Python reducer until
  a real workload demands it; the trigger to revisit is explicit (a CPU-infeasible target
  computation).
- The router threshold stays conservative (the GPU tier is not selected where it loses)
  and is revisited only after the future kernel exists and its crossover is measured.
- This is a speed/scale decision, deliberately separate from object scope (0009) and the
  faithful-compute and validation discipline (0006/0007), which continue to govern: the
  deferred kernel earns its place only by passing the same `== reference` agreement, never
  by being fast.
