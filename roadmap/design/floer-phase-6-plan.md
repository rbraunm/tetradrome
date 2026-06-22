# Phase 6 execution plan — native knot Floer

**Status:** in progress. This is the working plan for closing Phase 6 (native knot
Floer homology). It is governed by [0011](../decisions/0011-harden-the-brute-reference.md)
(exact-only acceleration, agreement discipline) and [0007](../decisions/0007-faithful-raw-path-no-heuristics.md)
(the raw grid complex is the reference), and feeds the engine tracker in
[homology-engine.md](homology-engine.md) §7.

## Where Phase 6 stands (audited 2026-06-08)

The engine is well past "just starting":

- It computes **HFK-hat**, the **three-genus** (`seifert_genus`), and **τ**, and validates
  all three against KnotInfo — `hfk_hat == ki.hfk_ranks` (up to mirror),
  `seifert_genus == three_genus`, `tau == ki.tau_invariant` (signed) — for a small roster
  (3_1, 4_1, 5_1, 5_2, plus 8_19 for τ; grid number up to 7).
- Structural tests confirm the differential grades correctly (Maslov −1, Alexander
  preserved) and squares to zero, on 3_1 and 4_1.
- The parallel / NUMA-pin / scaling paths agree with the serial reference on 3_1, 4_1,
  5_2 (closed in Phase 5).

Two real gaps:

- Floer is **not wired into the public `compute()`** — `test_invariants.py` explicitly
  marks `knot_floer_homology` as unsupported there.
- The **live `knot_floer_homology` calculator is not used as an oracle**. All validation
  is against KnotInfo's tabulated data; there is no independent second computation.

So Phase 6 is breadth + public-API integration + a second oracle + hardening, not building
the engine.

## What "Phase 6 done" means (acceptance)

1. HFK-hat (up to mirror), three-genus, and τ (signed) agree with **KnotInfo** across the
   tractable roster.
2. The same agree with the **independent `knot_floer_homology` calculator** on the roster
   — two independent oracles, not one table.
3. Floer is **first-class in `compute()`** and the result schema, with compute()-level tests.
4. Intractable grids **fail loud** via the [0008](../decisions/0008-memory-prediction-gate.md)
   memory gate — never a silent OOM like the staircase-11 sweep.
5. The agreement discipline (reference == accelerated) holds — already true from Phase 5.

## Work streams

### A — Validation breadth (the capability bar)

- **A1. Build the roster. [done 2026-06-08]** Of KnotInfo's 12,966 knots, 2,977 have HFK,
  τ, and three-genus all tabulated. Filtered by grid size (the tractability gate — n=11
  OOM'd at 200 GiB): 13 knots at n ≤ 8, 41 at n ≤ 9, **185 at n ≤ 10**; the remaining 2,792
  at n ≥ 11 are past the brute floor. The roster is derived (filter for data-present and
  n ≤ threshold), not hardcoded, so it tracks the table.
  - **Acceptance ceiling: n ≤ 10 (185 knots).** Two tiers: **n ≤ 8 (13)** is the routine
    tier — fast in the sandbox, run every iteration; **n ≤ 10 (185)** is the full
    acceptance sweep, run occasionally overnight on labradorite (the n=10 tier alone is
    144 knots at ~3.6M generators each). n ≥ 11 is gated out (0011/0008).
- **A2. Run roster agreement.** HFK-hat, genus, and τ (signed) vs KnotInfo across the
  roster. **Tier-0 (n ≤ 8) done [2026-06-08]: 13/13 exact** — direct equality on all three,
  no mirror resolution needed even for the off-table knots (6_1/2/3, 8_20/21, 9_42/46,
  10_124). Chirality (D1) did not bite at this tier. Remaining: the full n ≤ 10 sweep on
  labradorite via `pytest --heavy` (HFK runs across all cores; the flag refuses environments
  too small to run it), watching the n=9–10 knots for any chirality edge.
- **A3. Freeze** the passing roster as the Phase-6 acceptance set; record which knots are
  gated out by grid size and why (the irreducible-floor doctrine, 0011).

### B — Public API integration

- **B1.** Wire Floer into `compute()`: HFK-hat, τ, and three-genus as first-class
  invariants alongside Khovanov / Lee / Rasmussen *s*.
- **B2.** Settle the result-schema shape for the HFK bigraded rank table, signed τ, and
  genus, per SPEC §13.
- **B3.** Flip `test_invariants.py` from "not supported" to supported; add compute()-level
  Floer tests.

### C — Independent oracle cross-check

- **C1.** Wire the `knot_floer_homology` calculator as an optional validator backend (it
  returns HFK-hat — the same object `hfk_hat()` produces, so the comparison is direct).
- **C2.** Agreement test: `hfk_hat(grid)` vs the calculator across the roster — an
  independent computation, not KnotInfo's table.
- **C3.** Map the chirality/convention difference between our output and the calculator's.

### D — Hardening

- **D1.** Chirality/normalization. Direct equality (no mirror, correct τ sign) held across
  all 13 knots at n ≤ 8, so the KnotInfo `grid_notation` chirality appears to match the
  tabulated convention. D1 may reduce to confirming this persists through n = 10; if any
  knot disagrees by a mirror there, systematize the resolution rather than special-case it.
- **D2.** Verify the V-factor extraction (grid homology = HFK-hat ⊗ V^(n−1)) divides out
  correctly across all roster grid sizes, not just n ≤ 7.
- **D3.** Intractable grids must fail loud via the 0008 gate, with a test. (No silent OOM.)

## Sequence and dependencies

A1 first — it is the target. A2 runs with D1/D2 (chirality and the V-factor are what A2
surfaces). B proceeds once A2 confirms the engine output is correct. C is additive
confidence, after A2. D3 hooks the existing 0008 gate.

## Environment

- Streams A, B, D need only tetradrome + `database_knotinfo`. Confirmed running in the
  build sandbox (`hfk_hat(3_1)` matches KnotInfo), so the bulk of the work does not depend
  on a specific host.
- Stream C additionally needs the `knot_floer_homology` package installed (a heavier,
  compiled dependency); standing it up is a separate setup step taken when C begins.

## Decisions this rests on

- [0007](../decisions/0007-faithful-raw-path-no-heuristics.md) — the raw grid complex is
  the reference; only exact reductions accelerate it.
- [0008](../decisions/0008-memory-prediction-gate.md) — fail loud on intractable size
  (D3).
- [0011](../decisions/0011-harden-the-brute-reference.md) — the brute floor, the
  acceleration classes, and the binding-engine framing.
