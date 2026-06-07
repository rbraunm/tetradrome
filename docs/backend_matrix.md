# Backend Matrix

Which backend can produce which invariant (SPEC 12.3), corrected against what the
tools actually do in a plain pip environment (`roadmap/research/`). For the *names*
of invariants and how each backend spells them, see `conventions.md` and SPEC 12.4.

Legend: **available** (verified working), **planned** (intended, not built),
**sage-only** (method exists but requires Sage), **deferred** (behind a parked
spike), **n/a**, **future** (native, later).

| Canonical invariant | KnotInfo (oracle) | knot_floer_homology | Spherogram (pip) | Sage | KnotJob/JavaKh | native CPU | native CUDA |
|---|---|---|---|---|---|---|---|
| `alexander_polynomial` | available | n/a | sage-only | planned | n/a | future | n/a |
| `jones_polynomial` | available | n/a | sage-only | planned | possible | future | n/a |
| `signature` | available | n/a | sage-only (or via `seifert_matrix`) | planned | n/a | future | n/a |
| `determinant` | available | n/a | sage-only (or via `seifert_matrix`) | planned | n/a | future | n/a |
| `arf_invariant` | available | n/a | n/a | planned | n/a | future | n/a |
| `three_genus` | available | available | n/a | possible | n/a | n/a | n/a |
| `smooth_four_genus` | available | n/a | n/a | n/a | n/a | n/a | n/a |
| `topological_four_genus` | available | n/a | n/a | n/a | n/a | n/a | n/a |
| `rasmussen_invariant` | available | n/a | n/a | possible | deferred | future | future |
| `ozsvath_szabo_tau` | available | available | n/a | n/a | n/a | n/a | n/a |
| `epsilon` | available (HFK) | available | n/a | n/a | n/a | n/a | n/a |
| `nu` | available (HFK) | available | n/a | n/a | n/a | n/a | n/a |
| `fibered` | available | available | n/a | n/a | n/a | n/a | n/a |
| `l_space_knot` | available (HFK) | available | n/a | n/a | n/a | n/a | n/a |
| `khovanov_homology` | available (incl. mod-2) | n/a | n/a | possible | deferred | future (F2 first) | future |
| `knot_floer_homology` | available (HFK) | available (ranks/total_rank) | n/a | n/a | n/a | future | n/a |

## Standalone-pip reality (the practical picture)

- **Diagrams:** Spherogram, pip, standalone -- `PD_code()`, `DT_code()`,
  `seifert_matrix()`.
- **Floer side:** `knot_floer_homology`, pip, standalone -- the full HFK bundle.
- **Validation oracle:** KnotInfo via `database_knotinfo`, pip, standalone --
  including mod-2 Khovanov known answers.
- **Khovanov / Rasmussen *computation*:** no pip path. Requires Sage or KnotJob,
  both deferred. Until then, these invariants come from KnotInfo as known answers,
  and `signature`/`determinant` can be derived natively from Spherogram's
  `seifert_matrix()` if needed.

## Deferred columns

Sage and KnotJob/JavaKh are not yet stood up (`roadmap/milestones.md` "Deferred").
The matrix marks what they are expected to provide so the gap is explicit.
