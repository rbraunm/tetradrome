# CLAUDE.md -- Tetradrome

A reproducible, audit-friendly, Python-first workbench for the invariants that constrain
smooth 4-dimensional topology -- the Khovanov, Lee, and knot Floer homologies, the
Rasmussen *s*-invariant, tau, epsilon, nu, and the classical and concordance invariants --
for knots, links, and braids, on one normalized, fully-provenanced surface.

Authoritative design lives in `SPEC.md`, `roadmap/decisions/` (esp. 0006, 0007, 0009),
and `roadmap/design/homology-engine.md`. Those govern on any conflict with this file or
with older "orchestration" language surviving elsewhere in the tree.

## Core principles (these are the project, not preferences)
- **Native compute path (decision 0006).** Tetradrome computes each invariant with its own code. Existing tools (Spherogram/SnapPy, `knot_floer_homology`, KnotJob/Khoca, KnotInfo, SageMath) are opt-in validators and gold-master checks ONLY -- never a runtime dependency of a computation. Do not add an orchestration path that calls them to produce an answer.
- **No heuristics in the core (decision 0007).** Only exact, answer-preserving reductions may optimize a computation. The raw, faithful path is first-class and always runnable.
- **Never present an unvalidated number as a fact.** Every result carries method, version, conventions, raw output, and an explicit validation status. The reporter distinguishes computation, obstruction, and theorem reference.

## Acceleration: agreement discipline (non-negotiable)
The core is built to be accelerated (JIT, multi-core/NUMA, optional GPU) WITHOUT changing
its answers.
- Every acceleration tier MUST reproduce the reference result exactly -- bit-for-bit, validated independently per tier. A faster tier that disagrees is broken, not "approximate". `d^2 = 0` checks on native complexes hold regardless of tier.
- The GPU dispatch threshold is a calibratable knob, never a hidden default. VRAM-aware routing reads available VRAM rather than hardcoding a cutoff; the right threshold differs by card.

Per-tier agreement traps to guard against:
- JIT (Numba) reducer: Numba compiles to fixed-width ints and wraps silently on overflow, where the pure-Python reference uses arbitrary precision. F2 is unaffected (mod 2); on the multimodular Q path, ensure intermediate products (a*b before reduction) cannot overflow int64 under the chosen moduli, and that agreement tests include inputs large enough to expose overflow -- not just small cases.
- GPU dense kernel: arithmetic is exact (GF(2)/integer, no float), so any disagreement is from reduction order, races, or threshold routing -- not rounding. Agreement tests must exercise inputs on both sides of the VRAM dispatch threshold, so the GPU path and the CPU fallback are both covered and shown to agree.

## No silent fallbacks
If a requested GPU/JIT/NUMA path is unavailable, fail loudly with a clear message. Never
silently fall back to the reference path and report success -- that hides a broken
environment and produces misleading benchmarks.

## Hardware-dependent tiers
Exercise each only where its hardware exists; the agreement check must pass before any
benchmark number means anything.
- Numba JIT reducer -- any CPU.
- Batched dense GPU kernel -- a CUDA-capable GPU. Calibrate the VRAM threshold to the actual card; small-VRAM results are a checkpoint, not the final calibration for a larger card.
- NUMA-aware core pinning -- a multi-socket host; a no-op on single-socket machines.

Local validation order: baseline pytest; GPU detection + cupy install; JIT agreement
tests; NUMA pinning benchmarks.

## Testing
Tests make real assertions about computed invariants and complexes. No monkeypatching the
logic under test. Tests verify behavior; they never drive design.

## Python
Invoke as `python`, not `python3`.

---
Global standards (git workflow, fail-loud philosophy, no dead code, PowerShell rules) live
in the user-level `~/.claude/CLAUDE.md`. Transient session state and host-specific notes
live in `CLAUDE.local.md` (gitignored), not here.
