# Conventions

The fixed vocabulary and rules a result must follow. The goal is that no
backend-specific spelling or assumption leaks into a user-facing result
(SPEC 13.3).

## Knot identity

A knot's canonical id is its **KnotInfo name**: Hoste-Thistlethwaite (`K11n34`)
for 11 or more crossings -- the form both `spherogram.Link(...)` and Sage's
KnotInfo enum accept -- and Rolfsen (`3_1`, `4_1`, `10_124`) for 10 or fewer
(SPEC 12.4).

## Canonical invariant names

Defined and frozen in `roadmap/decisions/0001-canonical-invariant-names.md`. The
canonical name is chosen on the mathematics; each backend's spelling is mapped onto
it by the normalizer. The set: `alexander_polynomial`, `jones_polynomial`,
`signature`, `determinant`, `arf_invariant`, `three_genus`, `smooth_four_genus`,
`topological_four_genus`, `rasmussen_invariant`, `ozsvath_szabo_tau`, `epsilon`,
`nu`, `fibered`, `l_space_knot`, `khovanov_homology`, `knot_floer_homology`,
`smoothly_slice`, `topologically_slice`.

Confirmed backend-spelling mappings live in that decision record and in
`docs/backend_matrix.md` (notably KnotInfo's `ozsvath_szabo_tau_invariant` and the
Sage-gating of Spherogram's polynomial methods).

## Diagram notation

All four Spherogram / KnotInfo notations are first-class inputs and map
name-for-name (SPEC 12.4): PD code (list of 4-tuples) <-> `pd_notation`; DT code
<-> `dt_notation`; Gauss code <-> `gauss_notation`; braid word <-> `braid_notation`.
Spherogram's accessors are `PD_code()` and `DT_code()` (capitalized).

## Coefficient field and gradings

- The first native engine works over **F2**
  (`roadmap/decisions/0003-native-coefficient-field.md`).
- Grading conventions (homological / quantum degree, reduced vs unreduced,
  orientation and crossing-sign choices) must be recorded per backend in the
  result's `Provenance`, not assumed. A `ConventionMismatch` is raised rather than
  silently reconciling two conventions. [needs-review: pin the exact grading
  convention per backend once each adapter is built.]

## Parsing oracle data (no silent fallbacks)

Oracle values are strings and may be blank or carry a sentinel (KnotInfo `""` or
`"does not exist"`). These become an explicit `not_available`, never a coerced
default. A blank read as `0` is a wrong answer, not a missing one
(`research/knotinfo.md`,
`roadmap/decisions/0004-validate-by-default-error-policy.md`).

## Claim strength

Every result states its strength: `computation` (a number was produced),
`obstruction` (a property is obstructed), or `theorem_reference` (it rests on a
cited theorem). These are distinct from `proof`.

**Obstruction is not classification (SPEC 18).** A nonzero obstruction can prove a
knot lacks a property (e.g. not smoothly slice). A zero obstruction does **not**
prove the property holds. Reporting language must preserve this distinction.
