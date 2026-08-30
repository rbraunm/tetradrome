# 0004 - Validate-by-default and error policy

**Status:** Accepted.

## Context

The project's whole value is checkable results (README, "Validation and trust").
The house rule is no silent fallbacks: fail loudly and early rather than return a
plausible-looking wrong answer. SPEC 13.10 defines the public error set.

Validation is a setting rather than a flag because "validated" must not be able to
quietly mean "checked against a tabulated table, because that was all that happened to
be installed on this box." What a caller relied on has to be legible from what they
asked for.

## Decision

- `invariants.compute(...)`, `invariants.compute_all(...)`, and `export.build(...)`
  take an explicit `validate` **mode**, defaulting to **`"strict"`**:

  - **`strict`** (default): a computed oracle is required wherever one exists in the
    world. A missing required oracle or any oracle disagreement raises. KnotInfo rides
    along as an additional cross-check. The oracle requirement itself is decision 0006.
  - **`soft`**: use the computed oracle if installed, else fall back to KnotInfo with an
    info message saying the fallback would have been a strict-mode error. A computed
    oracle that runs and disagrees still raises -- soft tolerates absence, never
    mismatch.
  - **`off`**: no validation. The result still carries its `ValidationStatus`.

  There is no boolean form. Call sites pass an explicit mode, with no compatibility
  shim, per the house no-legacy-without-users rule.

- A result that has **no** known-answer oracle match and **no** independent
  cross-backend agreement raises **`UnvalidatedResult`** rather than being returned.
  `UnvalidatedResult` also covers a strict run whose required computed oracle is absent;
  its message points at `scripts/install_oracles.sh` for a provisioning gap, or names
  the not-yet-wired gap for a development one.

- The full loud error set (no silent coercion anywhere):
  `UnknownKnot`, `BackendUnavailable`, `UnvalidatedResult`, `ConventionMismatch`,
  `ExportHashMismatch`.

- Oracle data that is blank or carries a sentinel (e.g. KnotInfo `""` or
  `"does not exist"`) becomes an explicit `not_available`, never coerced to a
  default. The unknot row already shows the hazard: a blank `determinant` read as
  `0` would be a wrong answer, not a missing one (`research/knotinfo.md`).

## Consequences

- A caller must ask for `validate="off"` explicitly to receive an unvalidated value,
  and even then it carries its `ValidationStatus`.
- The mode a caller passes records what they relied on, so a validated result can never
  be confused with one that merely found nothing to contradict it.
- `export.load(...)` verifies the content hash and raises `ExportHashMismatch` on
  mismatch, so a consumer can trust a loaded roster without touching a backend.
- Parsing and normalization code never substitutes defaults for missing inputs.
