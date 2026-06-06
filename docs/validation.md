# Validation

Tetradrome is built for skepticism. An impressive-looking result with no
validation path is a bug, not a feature (README, "Validation and trust").

## The questions every result must answer (SPEC 14)

1. What invariant was requested?
2. What backend / code path produced it?
3. What backend version?
4. What conventions were assumed?
5. Was it checked against a known example?
6. Was it checked against an independent implementation or published table?
7. If native, was `d^2 = 0` verified?
8. What does the result prove?
9. What does it not prove?

The mantra: no topology output counts unless it passes boring, independent,
repeatable tests.

## Validation status

Each result carries a `ValidationStatus` with three checks (SPEC 11):

- `known_answer_match`: `pass` / `fail` / `not_available` -- agreement with an
  oracle (KnotInfo).
- `independent_backend_match`: `pass` / `fail` / `not_run` -- agreement with a
  second backend.
- `d_squared_check`: `pass` / `fail` / `not_applicable` -- for native complexes.

A result `is_validated` only if a known-answer or cross-backend check passed. Under
the validate-by-default policy, an unvalidated result raises `UnvalidatedResult`
rather than being returned silently
(`roadmap/decisions/0004-validate-by-default-error-policy.md`).

## What "validated" looks like in practice

The research notes already demonstrate the harness on small knots
(`roadmap/research/`):

- `4_1`: KnotInfo `three_genus` / `ozsvath_szabo_tau` / `fibered` agree with the
  `knot_floer_homology` backend -- oracle vs. independent backend, `pass`.
- `3_1`: KnotInfo `rasmussen_invariant = 2` and `ozsvath_szabo_tau = 1` satisfy
  `s = 2*tau` -- an internal consistency check.

## Computation vs obstruction vs proof

Reports separate what was **computed**, what is an **obstruction**, and what rests
on a **theorem reference** (SPEC 18). A nonzero obstruction can prove a knot lacks
a property; a zero obstruction proves nothing about the property holding. The
reporting language must never blur these into a bare "the knot is slice."
