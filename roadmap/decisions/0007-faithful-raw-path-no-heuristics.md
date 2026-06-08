# 0007 - Faithful raw path; exact reductions only; no heuristics in the core

**Status:** Accepted

## Context

Computing Khovanov / Floer is hard, and the fast tools in the field routinely bake in
assumptions — thinness, alternation, "mod 2 and assume no torsion" — that are wrong on
exactly the outlier knots this project exists to study. A shortcut that limits
use-cases is worthless for a workbench built to examine the exceptions.

The key distinction is that **not all "smart" algebra is a shortcut.** Delooping and
local Gaussian elimination (Bar-Natan's divide-and-conquer / "local" algorithm) are
*chain homotopy equivalences*: provably identical homology, gradings, and every
derived invariant, with no assumption about the knot. They are a faithful algorithm
choice ("Bareiss instead of cofactor expansion"), categorically different from a
heuristic.

## Decision

- The **raw, unreduced computation is first-class and always runnable**: the source of
  truth, the most general path (any coefficient ring, any derived quantity), and the
  reference every optimization is checked against. It may be slower and more
  memory-hungry; correctness and generality outrank speed.
- Only **exact, answer-preserving reductions** (homotopy equivalences: delooping,
  local elimination) are permitted as optimizations, and only as an **optional,
  toggleable** layer, verified `raw == reduced`.
- **No heuristics in the core:** no grading truncation, no thinness or alternation
  assumptions, no probabilistic / Monte-Carlo rank, no "mod 2 and assume no torsion,"
  no early-termination guesses.
- Acceleration tiers (JIT / NUMA / GPU) must return the identical answer as the
  reference — decision 0004 (validate by default) pointed inward.

## Consequences

- Answers are trustworthy on the inputs that matter most: the weird ones.
- Optimizations are safe by construction — an exact reduction cannot change an
  invariant, and the `raw == reduced` and `tier == reference` checks catch any
  implementation slip.
- We forgo the speed heuristics would buy. That is accepted deliberately: a fast wrong
  answer on a Conway-class knot is the precise failure mode the project exists to
  avoid.
- This constrains every future homology engine and the shared algebra back end
  (`roadmap/design/homology-engine.md`).
- Whether the reference path's *speed* is itself worth investing in — and the rule for
  classifying any proposed acceleration — is decided in 0011, which builds on this ADR.
