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

## Amendment (2026-07-05): validation, when relied upon, requires a computed oracle

**Status:** Accepted (amends the Decision above).

The original decision settled the *compute path*: no external tool produces an
answer. It left validation "opt-in," which in practice was read as "skip the
computed oracle whenever it is not installed on this box and quietly accept
KnotInfo instead." That is a silent fallback: it inherits every bug and
convention of one tabulated source and still calls the result validated.

Sharpened rule, separating *backing* a computation from *validating* one:

- **Backing** (producing the answer): still forbidden. kfh, SnapPy, Sage, etc.
  never compute a returned value. The compute-path decision above stands.
- **Validating** (checking the answer): when a consumer relies on validation, a
  computed oracle is **required wherever one exists in the world**. KnotInfo
  (tabulated) is a cross-check that rides along, and is the sole validator only
  for invariants nothing in the world computes.

`compute()` and the other validate-by-default entry points (ADR 0004) take three
explicit modes:

- **strict** (default): a computed oracle is required. A missing required oracle
  raises, pointing at `scripts/install_oracles.sh` (not installed) or naming the
  wiring gap (not yet wired into `compute()`). Any oracle disagreement raises.
  KnotInfo rides along as an additional cross-check.
- **soft**: use the computed oracle if installed; if none is installed, fall back
  to KnotInfo and surface an info message that the fallback was a consequence of
  soft mode and would have been an error in strict. An oracle that runs and
  disagrees still raises. soft tolerates absence, never a mismatch.
- **off**: no validation.

"Not installed on this box" is never grounds for silent KnotInfo substitution: it
is a provisioning gap (`install_oracles.sh`), or for oracles too heavy for the
sandbox (Sage, Khoca) a reason to run validation on CT 250 or other provisioned
compute.

### Reconciling with the original License and Portability arguments

Both are preserved. Tetradrome still ships pure-Python and GPL-free; it bundles
no oracle. kfh is separately installed by the user and imported only in the
validation path via the adapter API, never linked into a shipped combined work,
and the answer path never touches it. "Installs and runs anywhere Python runs"
holds for soft and off, which need no computed oracle. strict is the gold-standard
mode that additionally requires the world's computed oracle; where no kfh wheel
exists, strict fails loud with instructions and soft/off remain fully functional.
The requirement is opt-in by construction: it binds only the consumer who asks
for strict validation, which is that consumer's deliberate choice.
