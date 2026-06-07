# Claim Ledger

Living copy of SPEC Section 15. Every claim Tetradrome makes about its own
correctness lives here with a status and the evidence behind it. Update statuses
as evidence accrues; an impressive result with no validation path is a bug, not a
feature (README, "Validation and trust").

## Status legend

```
Red     = not implemented or not validated
Yellow  = partially validated / limited evidence
Green   = validated against multiple known cases
Blue    = independently reproduced or externally reviewed
```

## Ledger

Reorganized after the native-first pivot (ADRs 0006/0007/0009). The orchestration-era
claims about *external* compute backends (a Spherogram adapter, a `knot_floer_homology`
backend, "select a Khovanov/Rasmussen backend via Sage/KnotJob") are **superseded**:
external tools are validation oracles only, never a compute step. KnotInfo remains the
oracle. The claims below are about Tetradrome's own native computation.

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| KnotInfo oracle retrieves known-answer data | Green | `database_knotinfo` read layer; drives validate-by-default in `compute()` | det/sig/Alexander/Jones/Khovanov/s columns parsed |
| Classical invariants (determinant, signature, Alexander) computed natively | Green | Collins Seifert matrix from a braid word; validated vs KnotInfo at scale, incl. Conway Alexander = 1 | off-table torus knots cross-checked too |
| Jones polynomial computed natively | Green | Kauffman bracket over the resolution cube; validated vs KnotInfo through 11 crossings | warm-up for the cube machinery |
| Native unreduced Khovanov (F2) builds and is correct | Green | cube + enhanced states + differential; validated vs KnotInfo's mod-2 (UCT from the integral vector) across 8 knots, up to the documented mirror; unknot/trefoil explicit | |
| Native differential satisfies d² = 0 | Green | verified at every grading over **F2 and ℚ** for all tested knots; enforced in the reducers | the real correctness gate, esp. for the ℚ signs |
| Native rational Khovanov builds and is correct | Green | signed cube over ℚ; validated vs KnotInfo free ranks; signed-reduced-mod-2 reproduces the F2 answer | ℚ vs F2 differ by exactly the ℤ/2 torsion |
| Lee homology computed natively | Green | the deformed complex over ℚ; Lee's theorem (dim 2 for a knot) across 8 knots | |
| Rasmussen *s* computed natively | Green | read off the Lee quantum filtration; validated vs KnotInfo across s = 0, ±2, ±4, ±6 (incl. T(3,4)) | the invariant with concordance teeth |
| Outputs normalize into a shared, validated schema | Green | `InvariantResult` / `Provenance` / `ValidationStatus`; `compute()` validate-by-default, oracle mirror handled in one place | all of the above flow through `compute()` |
| Exact reductions preserve homology (`raw == reduced`) | Green | Gaussian cancellation (field-agnostic F2/ℚ); reproduces the rank-based homology per grading across the catalog and the bigraded F2 table; collapses the cube to its homology dimension | Phase 4; unoptimized reference, packed/accelerated cancellation is Phase 5 |
| Acceleration tiers agree with the reference | Red | not implemented | Phase 5 (packed-bit F2 / JIT / GPU) |
| Native Floer (τ, ε, ν, HFK ranks) | Red | not implemented | Phase 6 (peer engine) |
| Conway-adjacent workflow report is reproducible | Red | not assembled | needs the report generator + a curated input trail |
| Existing tooling survey completed | Yellow | initial list in SPEC §4 + `roadmap/research/` notes | needs maintenance / expert review |
