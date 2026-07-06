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

## Amendment (2026-07-05): validate is a three-mode setting

**Status:** Accepted (amends the Decision above).

The boolean `validate` is replaced by an explicit mode, so "validated" can no
longer quietly mean "checked against a tabulated table because that was all that
happened to be installed" (see the 0006 amendment):

- `validate="strict"` (default): a computed oracle is required wherever one
  exists in the world; a missing required oracle or any oracle disagreement
  raises. KnotInfo rides along as an additional cross-check. This is what the old
  `validate=True` becomes.
- `validate="soft"`: use the computed oracle if installed, else fall back to
  KnotInfo with an info message that the fallback would have been a strict-mode
  error. A computed oracle that runs and disagrees still raises.
- `validate="off"`: no validation. Replaces `validate=False`; the result still
  carries its `ValidationStatus`.

The loud error set, the `not_available` treatment of blank/sentinel oracle data,
and the export hash check are unchanged. `UnvalidatedResult` now also covers a
strict run whose required computed oracle is absent (its message points at
`scripts/install_oracles.sh`, or names the not-yet-wired gap). The boolean form
is not retained; call sites pass an explicit mode (no external-consumer
compatibility layer, per the house no-legacy-without-users rule).
