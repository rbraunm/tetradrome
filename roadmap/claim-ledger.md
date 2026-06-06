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

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| Existing tooling survey completed | Yellow | Initial list in SPEC 4; empirical verification of pip backends underway | Needs maintenance and expert review |
| Spherogram adapter handles initial knot catalog | Red | Not implemented | First engineering milestone |
| KnotInfo importer retrieves known-answer data | Red | Not implemented | Required for validation-first workflow |
| knot_floer_homology backend runs on small knots | Red | Not implemented | Floer v1 target |
| Khovanov/Rasmussen backend selected and callable | Red | Not implemented | Requires Sage / KnotJob / JavaKh evaluation |
| Backend outputs normalize into shared schema | Red | Not implemented | Required before reports are meaningful |
| Conway workflow report is reproducible | Red | Not implemented | Requires input, backend, validation, and report trail |
| Native mod-2 Khovanov complex builds for unknot/trefoil | Red | Deferred | Native v2/v3 feature |
| Native differential satisfies d^2 = 0 | Red | Deferred | Required for every native complex |
| GPU backend agrees with CPU backend | Red | Deferred | GPU optional until CPU is trusted |
