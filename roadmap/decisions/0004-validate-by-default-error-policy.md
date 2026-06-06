# 0004 - Validate-by-default and error policy

**Status:** Accepted.

## Context

The project's whole value is checkable results (README, "Validation and trust").
The house rule is no silent fallbacks: fail loudly and early rather than return a
plausible-looking wrong answer. SPEC 13.10 defines a public error set and a
`validate=True` default on the compute and export entry points.

## Decision

- `invariants.compute(...)`, `invariants.compute_all(...)`, and `export.build(...)`
  default to **`validate=True`**.
- With `validate=True`, a result that has **no** known-answer oracle match and
  **no** independent cross-backend agreement raises **`UnvalidatedResult`** rather
  than being returned.
- The full loud error set (no silent coercion anywhere):
  `UnknownKnot`, `BackendUnavailable`, `UnvalidatedResult`, `ConventionMismatch`,
  `ExportHashMismatch`.
- Oracle data that is blank or carries a sentinel (e.g. KnotInfo `""` or
  `"does not exist"`) becomes an explicit `not_available`, never coerced to a
  default. The unknot row already shows the hazard: a blank `determinant` read as
  `0` would be a wrong answer, not a missing one (`research/knotinfo.md`).

## Consequences

- A caller must opt out explicitly (`validate=False`) to receive an unvalidated
  value, and even then it carries its `ValidationStatus`.
- `export.load(...)` verifies the content hash and raises `ExportHashMismatch` on
  mismatch, so a consumer can trust a loaded roster without touching a backend.
- Parsing and normalization code never substitutes defaults for missing inputs.
