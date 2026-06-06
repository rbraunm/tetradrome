"""Invariants from the Seifert form, computed in exact arithmetic (no Sage, no
numpy). For a Seifert matrix V:

- knot determinant = |det(V + V^T)|
- knot signature   = signature of V + V^T (Sylvester's law of inertia)

Determinant uses the Bareiss algorithm (exact integer). Signature uses symmetric
congruence over the rationals (exact). Both are validated against KnotInfo
(tests/test_invariants.py).
"""
from __future__ import annotations

from fractions import Fraction

Matrix = list[list[int]]


def _symmetrize(v: Matrix) -> Matrix:
    n = len(v)
    return [[v[i][j] + v[j][i] for j in range(n)] for i in range(n)]


def integer_determinant(m: Matrix) -> int:
    """Exact determinant of an integer matrix via the Bareiss algorithm."""
    n = len(m)
    a = [row[:] for row in m]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if a[i][k] != 0), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // prev
        prev = a[k][k]
    return sign * a[n - 1][n - 1]


def _signature_symmetric(s: Matrix) -> int:
    """Signature (#positive - #negative eigenvalues) of a symmetric integer matrix,
    by exact symmetric congruence."""
    n = len(s)
    a = [[Fraction(s[i][j]) for j in range(n)] for i in range(n)]
    pos = neg = 0
    k = 0
    while k < n:
        if a[k][k] == 0:
            # Prefer swapping in a nonzero diagonal entry (symmetric row+col swap).
            piv = next((i for i in range(k + 1, n) if a[i][i] != 0), None)
            if piv is not None:
                a[k], a[piv] = a[piv], a[k]
                for r in range(n):
                    a[r][k], a[r][piv] = a[r][piv], a[r][k]
            else:
                # No nonzero diagonal; find an off-diagonal partner and add its
                # row+col into k to create a nonzero diagonal (2 * a[k][j]).
                j = next((j for j in range(k + 1, n) if a[k][j] != 0), None)
                if j is None:
                    k += 1  # fully isotropic remainder (degenerate); contributes 0
                    continue
                for c in range(n):
                    a[k][c] += a[j][c]
                for r in range(n):
                    a[r][k] += a[r][j]
        pivot = a[k][k]
        if pivot > 0:
            pos += 1
        else:
            neg += 1
        for i in range(k + 1, n):
            if a[i][k] != 0:
                f = a[i][k] / pivot
                for c in range(n):
                    a[i][c] -= f * a[k][c]
                for r in range(n):
                    a[r][i] -= f * a[r][k]
        k += 1
    return pos - neg


def determinant(seifert: Matrix) -> int:
    """The knot determinant |det(V + V^T)|."""
    return abs(integer_determinant(_symmetrize(seifert)))


def signature(seifert: Matrix) -> int:
    """The knot signature: signature of V + V^T."""
    return _signature_symmetric(_symmetrize(seifert))
