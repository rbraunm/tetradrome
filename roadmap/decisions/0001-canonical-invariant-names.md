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
attributed. The frozen v1 set:

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
