# Conventions

The fixed vocabulary and rules a result follows, so that no tool-specific spelling or
assumption reaches a user-facing result.

## Knot identity

A knot's canonical id is KnotInfo's spelling of its name: `3_1`, `10_124`, `11n_34`.
`knots.from_name` also accepts the Hoste-Thistlethwaite forms `K11n34` and `11n34` and
normalizes them. A diagram given by PD code or braid word with no name has no identity.

## Canonical invariant names

Defined in `roadmap/decisions/0001-canonical-invariant-names.md`. SPEC 7 carries the
same set beside each tool's spelling; `tests/test_canonical_names.py` keeps the two
and `invariants.compute()` in step.

## Diagram notation

KnotInfo's notations map name-for-name: PD code (list of 4-tuples) <-> `pd_notation`;
DT code <-> `dt_notation`; Gauss code <-> `gauss_notation`; braid word <->
`braid_notation`; grid diagram <-> `grid_notation`. Implemented inputs: name, PD code,
braid word. In a braid word `+j` means strand j crosses over strand j+1.

## Coefficient field, gradings, chirality

- F2 is the unmarked coefficient field; every other ring is named
  (`roadmap/decisions/0003-native-coefficient-field.md`, decision 0001).
- The native grading and chirality conventions, and how each compares with KnotInfo,
  are pinned in `roadmap/design/homology-engine.md` section 8.
- Each validator's convention against the native value (direct, mirrored, negated) is
  fixed in its adapter under `src/tetradrome/backends/` and verified by test, not
  assumed. Two conventions are never reconciled silently: a mismatch raises.

## Parsing oracle data

Oracle values are strings and may be blank or carry a sentinel (KnotInfo `""` or
`"does not exist"`). These are read as missing, never as a default. A blank read as
`0` is a wrong answer, not a missing one (`roadmap/research/knotinfo.md`, decision
0004).

## Claim strength

A reported claim states its strength: `computation` (a number was produced),
`obstruction` (a property is obstructed), or `theorem_reference` (it rests on a
cited theorem). None of these is `proof`.

**Obstruction is not classification.** A nonzero obstruction can prove a knot lacks a
property (e.g. not smoothly slice). A zero obstruction does not prove the property
holds. Reports never reduce this to a bare "the knot is slice."
