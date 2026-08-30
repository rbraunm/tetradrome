# 0001 - Canonical invariant names

**Status:** Accepted. Reversible -- the choice is localized to the normalizer
(SPEC 13.3 / 12.4); changing a canonical spelling is a normalizer edit, not a
cross-codebase change.

## Context

SPEC 12.4 leaves the exact canonical spelling of each invariant as an explicit
open decision. The canonical name is chosen on the mathematics, independent of any
tool; each backend's own spelling is mapped onto it by the normalizer. Where the
literature offers more than one proper name (e.g. `tau` vs `ozsvath_szabo_tau`,
`three_genus` vs `seifert_genus`), either is legitimate and we must simply pick
one. Empirical research (`roadmap/research/`) corrected two backend spellings that
the SPEC table had wrong or short.

## Decision

Canonical names are `lower_snake_case`, use the term a topologist recognizes from
the literature, and use the attributed full form where the invariant is standardly
attributed. The frozen v1 set (one entry renamed and one added since — see the
Amendment below; this table is retained as the original record):

| Canonical name | Notes |
|---|---|
| `alexander_polynomial` | |
| `jones_polynomial` | |
| `signature` | |
| `determinant` | |
| `arf_invariant` | |
| `three_genus` | the Seifert / 3-genus |
| `smooth_four_genus` | |
| `topological_four_genus` | |
| `rasmussen_invariant` | the `s` invariant |
| `ozsvath_szabo_tau` | the `tau` invariant |
| `epsilon` | |
| `nu` | |
| `fibered` | |
| `l_space_knot` | |
| `khovanov_homology` | |
| `knot_floer_homology` | |
| `smoothly_slice` | |
| `topologically_slice` | |

Backend spelling is owned by the normalizer. Empirically confirmed mappings that
differ from a naive guess:

- KnotInfo column `ozsvath_szabo_tau_invariant` -> canonical `ozsvath_szabo_tau`
  (the SPEC table's `ozsvath_szabo_tau` column name was wrong; see
  `research/knotinfo.md`).
- `knot_floer_homology` keys `tau` / `seifert_genus` -> `ozsvath_szabo_tau` /
  `three_genus`.
- Spherogram's polynomial methods exist but are Sage-gated, so they map under the
  Sage backend, not Spherogram-standalone (see `research/backends-pip.md`).

## Consequences

- One canonical vocabulary across `InvariantResult`, the roster export, and all
  reports. Consumers depend on these names, not on any backend's.
- The normalizer is the single place backend spellings live; adding a backend
  means extending the mapping, not touching the canonical set.
- Renaming a canonical term later touches only the normalizer and the docs.

## Amendment (2026-08-29): reconcile the frozen set with the shipped surface

**Status:** Accepted (amends the Decision above).

An audit against the code found the frozen v1 table and the names `compute()` actually
accepts had diverged. The table was never updated as the native engines landed, so a
reader following this ADR would write a call that raises. The authoritative set is the
`compute()` dispatch; this amendment brings the record onto it.

**Corrections to the v1 table:**

- `rasmussen_invariant` → **`rasmussen_s`**. The shipped name is `rasmussen_s`
  (21 call sites); `rasmussen_invariant` is rejected by `compute()`. This is a stated
  exception to the "attributed full form" rule above: the literature overwhelmingly
  writes the invariant as *s*, and `rasmussen_s` keeps the attribution while matching
  usage. `rasmussen_invariant` survives legitimately as the **KnotInfo column name** in
  `backends/knotinfo_backend.py` and its tests — that is backend spelling, owned by the
  normalizer, and is not affected.
- **`rational_khovanov_homology` is added** to the canonical set. It was in `compute()`
  and had 43 call sites while being absent from this record entirely.

**Coefficient field in canonical names.** The pair above encodes the field
asymmetrically: `khovanov_homology` means Khovanov over **F2**, and the rational lane
carries its own name. This was implicit and is now stated, because ADR 0003 defers
integral coefficients to a later engine and the next variant needs a rule rather than a
precedent to copy. The rule: **F2 is the unmarked default** (it is the first native
field, per 0003) and any other coefficient ring is named explicitly. A future integral
Khovanov is therefore `integral_khovanov_homology`, not a bare or suffixed variant.
The same rule governs the reduced theory when it lands
(`roadmap/design/homology-engine.md` §7, Phase 9).

**Comparison-layer row keys are not canonical names.** `scripts/comparison/spec.py`
uses `l_space` as a row key while the canonical name is `l_space_knot`, with
`adapters.py` mapping between them. That is legitimate — the comparison artifact has its
own presentation vocabulary — but it is a second namespace and is recorded here so the
mismatch is not read as drift. The rule: a name reaching `InvariantResult` or the roster
export is canonical; a name that only labels an artifact row is not.

**One claim in the Status line above is withdrawn.** It asserts that changing a
canonical spelling "is a normalizer edit, not a cross-codebase change." The
`rasmussen_s` case falsifies it: the name appears at 21 sites across `src/`, `scripts/`,
and `tests/`. Canonical renames are cross-codebase and should be costed as such. The
decision remains reversible; it is not as cheap to reverse as originally written.

**Consequence for the audit that produced this.** The divergence was invisible because
nothing checks the record against the code. A test pinning `compute()`'s supported set
against this table would have caught it, and is the natural follow-up.
