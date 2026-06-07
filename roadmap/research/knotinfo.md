# Research: KnotInfo validation oracle (database_knotinfo)

Empirical verification of the SPEC 12.2 priority-2 backend: the KnotInfo
known-answer source. Findings from installing and querying it in a clean pip
environment. Feeds the KnotInfo importer (Milestone 3), `catalog/known_answers.yaml`,
and `docs/backend_matrix.md`.

## Package

- `database_knotinfo` **2026.6.1**; install: `pip install database_knotinfo` (clean).
- License: **GPL**. Home: https://github.com/soehms/database_knotinfo (this is the
  same offline KnotInfo data package Sage uses).
- This is **data**, GPL-licensed, consumed as an optional pip dependency. Querying
  it is consistent with the SPEC 20 / README posture: we read it at authoring time
  and do not copy its rows into the repo. Confirm this stays true when we build the
  importer -- export validated values into our own `known_answers.yaml` with a
  `source: knotinfo` provenance tag rather than vendoring the table.

## Shape of the data

- `database_knotinfo.link_list()` -> a `list` of `dict`, **12,967 entries**
  (knots and links combined), each with **244 columns**.
- Every real column has an `_anon` twin (chirality-anonymized variant). Use the
  plain column; do not pull `_anon` by accident.
- **There is a descriptor row mixed in** (its `category` value is the literal
  string `"Category"`, names are column descriptions). The importer must filter it
  out -- skip rows where `name == "Name"` / `category == "Category"`.
- `category` for knots is the crossing number as a string (`"0".."12"`). Links
  are interleaved in the same list; filter to knots explicitly.
- All values are **strings** and need parsing: `fibered` is `"Y"`/`"N"`,
  polynomials are strings (note odd spacing, e.g. `"t+ t^3-t^4"`), integers are
  strings, missing values are blank `""`, and some are the sentinel
  `"does not exist"` (e.g. `positive_braid_notation` for `4_1`).

## Confirmed known-answer values (the point of the oracle)

| name | rasmussen_invariant | ozsvath_szabo_tau_invariant | three_genus | smooth_four_genus | signature | determinant | fibered |
|---|---|---|---|---|---|---|---|
| `0_1` (unknot) | (blank) | (blank) | 0 | 0 | 0 | (blank, shows 0) | (blank) |
| `3_1` | 2 | 1 | 1 | 1 | -2 | 3 | Y |
| `4_1` | 0 | 0 | 1 | 1 | 0 | 5 | Y |

## Cross-source agreement (validation harness, demonstrated)

- `4_1`: KnotInfo `three_genus=1`, `ozsvath_szabo_tau_invariant=0`, `fibered=Y`
  agree with the `knot_floer_homology` backend output from `backends-pip.md`
  (`seifert_genus=1`, `tau=0`, `fibered=True`). Independent oracle vs. backend agree.
- `3_1`: KnotInfo `rasmussen_invariant=2`, `ozsvath_szabo_tau_invariant=1` satisfy
  the expected `s = 2*tau` relation. Internal consistency check passes.

This is exactly the cross-check the project exists to automate.

## Native-engine oracle (bigger than expected)

KnotInfo ships **mod-2 Khovanov** data directly:
`khovanov_reduced_mod2_polynomial` / `_vector`, plus unreduced / integral /
rational / odd variants. This is the known-answer source for the native mod-2
Khovanov engine (Milestone 7) and for any KnotJob cross-check -- we can validate a
native `F2` complex against KnotInfo without needing Sage or KnotJob first.

## Corrections / refinements to SPEC 12.4

1. **tau column name is wrong.** The table maps canonical `ozsvath_szabo_tau` ->
   KnotInfo column `ozsvath_szabo_tau`. The real column is
   **`ozsvath_szabo_tau_invariant`**. (`rasmussen_invariant` is correct as written.)
2. Most other canonical names match real KnotInfo columns
   (`rasmussen_invariant`, `three_genus`, `smooth_four_genus`,
   `topological_four_genus`, `signature`, `determinant`, `jones_polynomial`,
   `alexander_polynomial`, `arf_invariant`, `fibered`, `pd_notation`,
   `dt_notation`, `gauss_notation`, `seifert_matrix`). The normalizer is mostly a
   rename, as the SPEC claims -- with the tau fix above.

## Implications for the build

- **Milestone 3 is unblocked and pip-only.** The KnotInfo importer reads
  `link_list()`, filters the descriptor row and links, maps columns to canonical
  names (SPEC 12.4, tau-corrected), and writes validated values into
  `catalog/known_answers.yaml`.
- **No silent fallbacks (house rule).** Blank and `"does not exist"` must become
  an explicit `not_available`, never coerced. The unknot row already shows the
  hazard: blank `determinant`/`signature` read back as `0`, which is a *wrong
  answer*, not a missing one. Parse blanks as missing, loudly.
- `known_answers.yaml` schema should carry, per value: canonical invariant name,
  parsed value, the raw KnotInfo string, and `source: knotinfo` + package version
  for provenance.

## Open items

- Decide knot-vs-link filtering rule precisely (category numeric vs link naming).
- Polynomial string parsing (variable conventions, spacing) for any value we want
  as structured data rather than an opaque string.
- Sage feasibility and KnotJob CLI spikes remain the gate for *computing*
  Khovanov/Rasmussen (vs. looking them up here). [research]
