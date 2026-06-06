"""The Jones polynomial, via the Kauffman bracket over the resolution cube.

State sum: <L> = sum over states s of  A^(a(s)-b(s)) * delta^(|s|-1),  where a/b are
the counts of A-/B-smoothings, |s| the number of circles, and delta = -A^2 - A^-2.
Then V_L(t) = (-A^3)^(-w) <L> with A = t^(1/4), where w is the writhe.

Conventions (A-smoothing, substitution direction, writhe sign) are pinned to KnotInfo
empirically: this reproduces KnotInfo's Jones polynomial -- chirality included -- for
every knot through 11 crossings.

The polynomial is returned canonically as `(low_exponent, coeffs)` with coeffs the
integer coefficients ascending from t^low_exponent and no leading/trailing zeros. The
Jones polynomial is a genuine invariant (V(unknot) = 1), so there is no unit ambiguity
to normalize away -- the representation is exact.

This computation is exponential in the crossing number (the cube has 2^n states); it
is the Phase 0 warm-up that exercises the cube skeleton, validated on small knots, not
a large-knot engine.
"""
from __future__ import annotations

from ..diagrams import seifert_structure
from ..diagrams.model import PDCode
from ..engines import cube

# Laurent polynomials in A as {exponent: coefficient}, zero coefficients dropped.
_ALaurent = dict[int, int]
_DELTA: _ALaurent = {2: -1, -2: -1}  # -A^2 - A^-2


def _add(p: _ALaurent, q: _ALaurent) -> _ALaurent:
    r = dict(p)
    for e, c in q.items():
        r[e] = r.get(e, 0) + c
        if r[e] == 0:
            del r[e]
    return r


def _mul(p: _ALaurent, q: _ALaurent) -> _ALaurent:
    r: _ALaurent = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            r[e1 + e2] = r.get(e1 + e2, 0) + c1 * c2
    return {e: c for e, c in r.items() if c}


def _pow(p: _ALaurent, k: int) -> _ALaurent:
    r: _ALaurent = {0: 1}
    for _ in range(k):
        r = _mul(r, p)
    return r


def kauffman_bracket(pd: PDCode) -> _ALaurent:
    """The (unnormalized-by-writhe) Kauffman bracket as a Laurent polynomial in A."""
    n = len(pd)
    total: _ALaurent = {}
    for state in cube.states(n):
        circles = cube.circle_count(pd, state)
        b = sum(state)              # number of B-smoothings
        sigma = (n - b) - b         # a - b
        total = _add(total, _mul({sigma: 1}, _pow(_DELTA, circles - 1)))
    return total


def canonical_laurent(low: int, coeffs) -> tuple[int, tuple[int, ...]]:
    """Strip leading/trailing zeros from `coeffs` (ascending from t^low), adjusting
    `low`. Returns (low, coeffs); the zero polynomial returns (0, ())."""
    c = [int(x) for x in coeffs]
    i = 0
    while i < len(c) and c[i] == 0:
        i += 1
        low += 1
    j = len(c)
    while j > i and c[j - 1] == 0:
        j -= 1
    return (low, tuple(c[i:j]))


def jones_polynomial(pd: PDCode) -> tuple[int, tuple[int, ...]]:
    """Jones polynomial of the knot given by `pd`, as (low_exponent, coeffs) in t."""
    if not pd:
        return (0, (1,))  # unknot: V = 1

    bracket = kauffman_bracket(pd)
    w = seifert_structure(pd).writhe

    # f = (-A^3)^(-w) <L> = (-1)^w * A^(-3w) * <L>.
    sign = -1 if w % 2 else 1
    f: _ALaurent = {}
    for a_exp, coeff in bracket.items():
        e = a_exp - 3 * w
        f[e] = f.get(e, 0) + sign * coeff

    # Substitute A = t^(1/4): A^e -> t^(e/4). For a knot every exponent is a multiple
    # of 4 (integer t-powers); anything else means a non-knot input or a bug.
    by_t: dict[int, int] = {}
    for a_exp, coeff in f.items():
        if a_exp % 4 != 0:
            raise ArithmeticError(
                f"Jones exponent {a_exp} is not a multiple of 4 (input is not a knot?)."
            )
        t_exp = a_exp // 4
        by_t[t_exp] = by_t.get(t_exp, 0) + coeff

    by_t = {e: c for e, c in by_t.items() if c}
    if not by_t:
        return (0, ())
    low = min(by_t)
    high = max(by_t)
    return canonical_laurent(low, [by_t.get(e, 0) for e in range(low, high + 1)])
