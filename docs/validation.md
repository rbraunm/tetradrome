# Validation

A result with no validation path is a bug.

## The questions every result must answer

1. What invariant was requested?
2. What code path produced it?
3. What version?
4. What conventions were assumed?
5. Was it checked against a known example?
6. Was it checked against an independent implementation or a published table?
7. If native, was `d^2 = 0` verified?
8. What does the result prove?
9. What does it not prove?

## Where the rules are

| Rule | Where |
|---|---|
| The three modes (`strict`, `soft`, `off`) and the errors they raise | `SPEC.md` section 6 |
| A computed oracle is required wherever one exists; KnotInfo alone validates only where none does | `SPEC.md` section 6 |
| Every validator's identity, version and verdict is recorded | `SPEC.md` section 5 |
| Which validators are wired, and which computed oracles exist but are not | `src/tetradrome/backends/registry.py` |
| What each oracle computes | `backend_matrix.md` |
| Provisioning the oracles | `scripts/install_oracles.sh` |
| Internal checks: `d^2 = 0`, `raw == reduced`, tier `==` reference, scheduled `==` serial | `roadmap/design/homology-engine.md` section 4 |
| Claim strength: computation, obstruction, theorem reference | `conventions.md` |

## Limits of the evidence

- On every KnotInfo row s = 2 tau, so agreement on `rasmussen_s` over the tabulated
  knots does not tell s from 2 tau, or s over Q from s over F2. A check that does needs
  an off-table knot such as 14n19265 (s over Q is 0, s over F2 is -2; arXiv:2610.37595,
  Table 2).
- KnotInfo's values change between releases (374 `unknotting_number` entries between
  2026.6.1 and 2026.11.2), which is why each result records the KnotInfo version it was
  compared with.
