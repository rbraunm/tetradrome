# 0006 - No external compute backends in the core

**Status:** Accepted

## Context

The heavy invariants (Khovanov, Lee / Rasmussen *s*, knot Floer) have mature external
implementations — most temptingly the `knot_floer_homology` package, a wrapper around
Zoltán Szabó's HFK Calculator. Depending on one would be the fast way to get τ/ε/ν or
*s*.

Four independent facts argue against it as a core dependency, any one sufficient:

- **License.** It is GPLv2+; Tetradrome is Apache-2.0. A GPL runtime dependency is a
  copyleft entanglement — shipped as a combined work, GPL terms could reach our code.
- **Portability.** It ships binary wheels only, with no source distribution, each a
  glibc/arch-pinned C++ extension. On any platform/Python without a prebuilt wheel it
  is *uninstallable* — there is nothing to build from. That fails "pure Python, runs
  anywhere."
- **Mission.** Using it reports someone else's answer, not one we computed. The
  workbench exists to compute the math.
- **Opportunity.** The compute kernel (sparse linear algebra over a cube of
  resolutions) is exactly what we want to own and accelerate (JIT / NUMA / GPU), not
  rent as an opaque binary.

`SPEC.md` §20 already restricts GPL tools to external-validator status; this ADR
generalizes that to all compute.

## Decision

The core computes its invariants itself. **No external library is a runtime
dependency of the compute path.**

External tools (SnapPy / spherogram, Szabó's HFK calculator, KnotJob, SageMath, etc.)
may be used **only as opt-in validators / cross-checkers**, the way KnotInfo is used —
never as a compute step, never bundled, never required to obtain an answer. Such
integrations live behind the `SPEC.md` §13.8 adapter contract, isolated and clearly
labeled as validation.

This holds regardless of an external tool's quality or maintenance status; the
deciding axes are portability, license, and "our math," not how good the tool is.

## Consequences

- Tetradrome installs and runs anywhere Python runs — no compiler, no GPL, no
  binary-wheel platform lottery.
- We carry the cost of implementing Khovanov, Lee, and Floer ourselves. That is the
  point, and it is what makes the auditability and the JIT/NUMA/GPU acceleration story
  possible at all.
- A maintained, well-built external tool can still be wired in later as an optional
  cross-checker; this ADR forbids it as *core compute*, not as validation.
