# 0001 - Canonical invariant names

**Status:** Accepted. Reversible, but not cheaply -- a canonical name appears across
`src/`, `scripts/`, and `tests/` (the `rasmussen_s` spelling alone spans 21 sites), so a
rename is a cross-codebase change, not a normalizer edit.

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
attributed. The canonical set:

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
| `rasmussen_s` | the `s` invariant; see the attribution note below |
| `ozsvath_szabo_tau` | the `tau` invariant |
| `epsilon` | |
| `nu` | |
| `fibered` | |
| `l_space_knot` | |
| `khovanov_homology` | Khovanov over F2 (the unmarked default; see below) |
| `rational_khovanov_homology` | Khovanov over Q |
| `knot_floer_homology` | |
| `smoothly_slice` | |
| `topologically_slice` | |

`invariants.compute()` is the authoritative surface: a name is canonical if and only
if `compute()` accepts it or the table above reserves it for an engine not yet built.

**Attribution and the `s` invariant.** `rasmussen_s` is a deliberate exception to the
"attributed full form" rule: the literature overwhelmingly writes the invariant as *s*,
and this spelling keeps the attribution while matching how it is actually referred to.
`rasmussen_invariant` is the **KnotInfo column name**, not a canonical name; it lives in
`backends/knotinfo_backend.py` as backend spelling and is owned by the normalizer.

**Coefficient field.** F2 is the unmarked default, because it is the first native field
(ADR 0003); every other coefficient ring is named explicitly. So `khovanov_homology`
means Khovanov over F2 and `rational_khovanov_homology` names the Q lane. A future
integral engine is `integral_khovanov_homology`, and the reduced theory
(`roadmap/design/homology-engine.md` section 7, Phase 9) follows the same rule.

**Comparison-layer row keys are not canonical names.** `scripts/comparison/spec.py`
uses its own presentation vocabulary for artifact rows -- `l_space` there against
canonical `l_space_knot`, mapped in `adapters.py`. A name reaching `InvariantResult` or
the roster export is canonical; a name that only labels an artifact row is not.

Backend spelling is owned by the normalizer. Empirically confirmed mappings that
differ from a naive guess:

- KnotInfo column `ozsvath_szabo_tau_invariant` -> canonical `ozsvath_szabo_tau`
  (the SPEC table's `ozsvath_szabo_tau` column name was wrong; see
  `research/knotinfo.md`).
- KnotInfo column `rasmussen_invariant` -> canonical `rasmussen_s`.
- `knot_floer_homology` keys `tau` / `seifert_genus` -> `ozsvath_szabo_tau` /
  `three_genus`.
- Spherogram's polynomial methods exist but are Sage-gated, so they map under the
  Sage backend, not Spherogram-standalone (see `research/backends-pip.md`).

## Consequences

- One canonical vocabulary across `InvariantResult`, the roster export, and all
  reports. Consumers depend on these names, not on any backend's.
- The normalizer is the single place backend spellings live; adding a backend
  means extending the mapping, not touching the canonical set.
- Renaming a canonical term is a cross-codebase change and is costed as one.
- Nothing currently checks this table against `compute()`, so the two can drift
  silently. A test pinning the supported set against this record is the guard.
