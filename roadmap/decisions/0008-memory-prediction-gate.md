# 0008 - Memory-prediction gate; fail loud and early; no silent shrink-to-fit

**Status:** Accepted

## Context

The hard cases are often memory-bound: fill-in during elimination blows up the
working set, and on a limited GPU the VRAM is the tight constraint. Launching a
computation that will not fit produces either an out-of-memory crash deep in a run or
a silent swap-death — both violate the project's fail-loud-early stance (and the
spirit of decision 0004).

What is predictable, honestly: the **initial complex size is exact and cheap** —
the dimension of every graded piece is combinatorial, computable from the diagram
without building anything (O(n·2ⁿ) for Khovanov). The **elimination peak (fill-in) is
boundable and estimable but not exact** — the classic sparse-direct unknown.

## Decision

Before a heavy computation:

- Estimate its memory and compare against available **VRAM specifically** (the tight
  constraint, far smaller than system RAM) and system RAM.
- Route on the result: GPU tier if it fits VRAM; CPU/RAM tier if it fits RAM but not
  VRAM; tile/stream if necessary; otherwise **refuse loudly with the number**
  ("needs ~X GB, you have Y") *before* starting. Never silently degrade to swap, never
  OOM mid-run.
- **Never silently shrink the math to fit.** Exact reductions (decision 0007) may be
  used as a *size* tool when the faithful raw path will not fit, but only as an
  explicit, user-chosen opt-in — never an automatic shrink-to-fit on constrained
  hardware.

Gating uses the exact complex-size figure as a hard floor and the fill-in bound +
estimate above it.

## Consequences

- A user on a small GPU gets a clear, up-front answer about what will and will not
  run — not a crash and not a thrash.
- This is the fail-loud-early rule applied to memory; it pairs with 0007 (the exact
  reduction is the only sanctioned way to make something fit, and only on request).
- The fill-in estimate is approximate, so the gate is conservative by construction
  (bound + estimate, exact floor); a rejected job is never a wrong answer, only a
  refused one.
