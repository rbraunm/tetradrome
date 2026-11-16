# Claim Ledger

Every claim Tetradrome makes about its own correctness, with its status and the evidence
behind it. A result with no validation path is a bug.

## Status legend

```
Red     = not implemented or not validated
Yellow  = partially validated / limited evidence
Green   = validated against multiple known cases
Blue    = independently reproduced or externally reviewed
```

## Ledger

Phases are those of `design/homology-engine.md` section 7.

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| KnotInfo oracle retrieves known-answer data | Green | `database_knotinfo` read layer; drives validation in `compute()` | determinant, signature, Alexander, Jones, Khovanov, s, HFK, τ, genus columns parsed |
| Classical invariants (determinant, signature, Alexander) computed natively | Green | Collins Seifert matrix from a braid word; validated against KnotInfo across the tables, including Conway Alexander = 1 | off-table torus knots cross-checked too |
| Jones polynomial computed natively | Green | Kauffman bracket over the resolution cube; validated against KnotInfo through 11 crossings | |
| Native unreduced Khovanov (F2) is correct | Green | cube + enhanced states + differential; validated against KnotInfo's mod-2 ranks (universal coefficients from the integral vector) across 8 knots, up to the documented mirror; unknot and trefoil explicit | |
| Native differential satisfies d² = 0 | Green | verified at every grading over F2 and ℚ for all tested knots; enforced in the reducers | the correctness gate for the ℚ signs |
| Native rational Khovanov is correct | Green | signed cube over ℚ; validated against KnotInfo free ranks; the signed complex reduced mod 2 reproduces the F2 answer | |
| Lee homology computed natively | Green | the deformed complex over ℚ; Lee's theorem (dimension 2 for a knot) across 8 knots | |
| Rasmussen *s* computed natively | Green | read off the Lee quantum filtration; validated against KnotInfo across s = 0, ±2, ±4, ±6 (including T(3,4)) | On every KnotInfo row s = 2τ, so this evidence cannot tell s from 2τ, or s over ℚ from s over F2. Open: a check on an off-table knot where they differ, such as 14n19265 (s over ℚ is 0, s over F2 is −2; arXiv:2610.37595, Table 2) |
| Results carry provenance and validation in one schema | Green | `InvariantResult` / `Provenance` / `ValidationStatus`; every invariant goes through `compute()`; the KnotInfo mirror is handled in one place | |
| Exact reductions preserve homology (`raw == reduced`) | Green | Gaussian cancellation over F2 and ℚ reproduces the rank-based homology per grading across the catalog and the bigraded F2 table | Phase 4; cancels after the full cube is built |
| Reduction tiers agree with the reference | Green | every tier reproduces the reference homology across the catalog; the GPU dense kernel and the Numba reducer confirmed on real hardware | `bitint` (pure Python) is the fastest in every tested regime; the on-device GPU kernel waits for a calculation the CPU tiers cannot finish |
| Scheduled calculations agree with their serial oracles | Green | the scheduled grading histogram, HFK-hat and τ are tested against `grading_histogram_reference`, `grid_complexes` and `tau_reference` | `design/scheduler.md` |
| Native Floer (HFK ranks, three-genus, τ, ε, ν) | Yellow | the grid engine computes HFK-hat, the three-genus and τ in `compute()`, tested against KnotInfo over the roster with no mirror or sign change (default tests: grid number ≤ 8, 13 knots; `pytest --heavy`: ≤ 10, 185 knots) and under strict against `knot_floer_homology` on 3_1, 4_1 and 8_19 | ε, ν not implemented (Phase 7); grid number 11 does not fit in memory |
| Auditable filtered Floer complex (CFK presentation) | Red | planned (Phase 7) | the bifiltered grid complex and a reduced model (staircase ⊕ boxes) as a serializable artifact; ε, ν, Υ read off it. `knot_floer_homology` (`complex=True`, since 1.1) returns the simplified complex over F[U,V]/(UV) and is an oracle for the reduced model |
| Conway-adjacent workflow report is reproducible | Red | not assembled | needs the report generator and a checked encoding of K′ |
| Existing tooling survey is current | Yellow | `docs/backend_matrix.md`, `docs/existing_tools.md`, `roadmap/research/` | needs expert review |
