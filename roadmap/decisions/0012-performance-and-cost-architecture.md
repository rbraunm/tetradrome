# 0012 - Performance and cost architecture: name the axes, one cost model, dissolve scaling.py

**Status:** Accepted

## Context

The reference path grew a performance layer across the Phase 5 and Phase 6 work --
parallel generation, parallel reduction, a memory predictor, the `--heavy` test tier --
and a read of the tree from a distance finds several modules whose names brush against
"parallel", "scaling", "performance", and "memory". That raises a fair question: is this
murkiness in the spec, drift during implementation, or a distinct-concerns map that only
looks redundant?

An audit settles it. The backend layer is coherent and already governed (0006, 0010):
`tiers.f2_homology` is one dispatcher over several reduction *algorithms* (reference,
bitint, packed words/dense, jit, gaussian, rational), each in its own `reduce_*` module --
a strategy pattern, not duplication. What is genuinely off is twofold. First, there are
**two cost models for the same physical quantity**: `algebra/memory.predict_size`
(reduction peak from a built complex, pivot-inclusive) and
`engines/floer/scaling.dense_reduction_bytes` (the same peak from a grading histogram,
pivot-omitting), wired to different callers and disagreeing on the formula. Second,
`engines/floer/scaling.py` is a grab-bag -- it holds the synthetic staircase grid,
generation parallelism, and the cost model under one vague name -- and generation itself
is split (serial `grid_complexes` in `homology.py`, parallel `parallel_grid_complexes` in
`scaling.py`) behind a mutual lazy import. No single record names the performance axes, so
two legitimately different parallelisms (generation vs reduction) read as possible
redundancy. This is drift plus a signposting gap, not a spec defect.

The pivot disagreement was resolved against the code, not assumed. `f2_rank_words`
allocates the packed matrix (`cols x nwords` uint64) **and** a `pivots` map holding up to
`min(cols, rows)` reduced columns. So `predict_size`'s `(cols + min(cols, rows)) * nwords
* 8` matches the real allocation; `dense_reduction_bytes` omits the pivot term and
under-counts the reduction peak by up to ~2x, which makes the memory guard built on it
looser than it appears.

## Decision

- **Name four orthogonal performance axes** and keep each to one home, so future work
  extends an axis rather than re-deriving one:
  1. **Backend tier** -- which algorithm reduces one complex. Home: `algebra/tiers.py`
     plus the `reduce_*` modules; selected by `route_backend`. (Already coherent; named
     here for completeness.)
  2. **Generation parallelism** -- splitting the n! generation across processes.
     Memory-cheap (peak is the complex, not the dense matrices).
  3. **Reduction parallelism + memory budget** -- reducing independent complexes (a
     knot's gradings, or knots in a batch) across processes, bounded by a RAM budget via
     deterministic waves. Home: `algebra/parallel.py`.
  4. **Reduction cost model** -- one predictor of reduction memory. Home:
     `algebra/memory.py`.

- **One cost model.** The per-block reduction cost is `(cols + min(cols, rows)) * nwords *
  8` -- matrix plus pivots -- verified against `f2_rank_words`. A single primitive
  computes it; `predict_size` and the histogram-based predictor both call it.
  `dense_reduction_bytes` is corrected onto this formula (it gains the pivot term). The
  model exposes two named aggregations: the **per-complex peak** (max over a grading's
  degrees -- the per-unit price the scheduler packs against and the single-unit
  feasibility test) and the **worst-case co-resident sum** (all gradings at once -- the
  unbounded-concurrency figure, informational only -- the feasibility criterion is the
  per-complex peak, because the scheduler runs the remainder in memory-bounded waves).

- **Dissolve the grab-bag.** Generation (serial and parallel) is colocated in one engine
  module, ending the mutual lazy import. The synthetic staircase grid moves to the grid
  constructor's home. The histogram enumerator stays engine-side (it needs the grid); the
  byte arithmetic moves to `algebra/memory.py`. `scaling.py` is deleted once emptied -- no
  dead modules.

These moves change no computed answer; the agreement tests must stay green across every
step, which is the migration's correctness gate.

## Consequences

- The memory guard's feasibility test is the **largest single grading** against the
  budget: a knot is infeasible (fail loud) only when that exceeds it, since the scheduler
  runs the remainder in memory-bounded waves. Wiring the guard to the per-complex-peak
  criterion is a follow-up.
- Because the OOM fence carries the pivot term, it does not under-count, and any
  projection is read off the pivot-inclusive formula. The size profile is negligible
  through n=9, dominant by n=10, and past any single machine by n=11.
- Reversible. This is an organizational decision (SPEC names no module layout); it can be
  rearranged again if a later engine wants a different cut. What is not reversible by fiat
  is the agreement discipline the moves run under.
