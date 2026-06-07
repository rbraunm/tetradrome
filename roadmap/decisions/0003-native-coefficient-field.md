# 0003 - Native coefficient field

**Status:** Accepted for the first native engine. Extensible later.

## Context

The native Khovanov construction (SPEC 13.4) and the exact-algebra layer
(SPEC 13.6) need a coefficient field. Working over the integers or the rationals
from the start drags in sign conventions and torsion bookkeeping that are easy to
get subtly wrong before any validation harness exists.

## Decision

The first native engine works over **F2 (mod 2)**. Integral and rational
coefficients are deferred and added later, behind the validation harness, only when
the F2 path is trusted.

## Consequences

- The first native differential is built and `d^2 = 0` is checked over F2, avoiding
  sign and torsion complexity (SPEC 13.4).
- KnotInfo supplies **mod-2 reduced/unreduced Khovanov** polynomials and vectors
  directly (`research/knotinfo.md`), so the F2 engine has a ready known-answer
  oracle without needing Sage or KnotJob first.
- The algebra layer's matrix type is packed-bit GF(2) (SPEC 13.6 / 13.7); a later
  integral/rational backend is an addition, not a rewrite.
