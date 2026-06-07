"""Graded cochain complexes over Q -- the generic-field lane of the back end.

GradedComplex (F2) is the fast lane for Khovanov mod 2 (decision 0003, the bit-set
GF(2) representation foreshadowed there). This is its rational counterpart, sitting
beside it behind the same conceptual interface rather than as a parallel stack -- the
field-tested pattern keeps an F2 fast path next to a general-field one (Ripser defaults
to F2 with prime-field support opt-in; PHAT separates the column representation from the
reduction algorithm) instead of forcing one representation to serve both. It is what Lee
homology and the Rasmussen s-invariant need: those are defined only over Q, and F2
Khovanov is Conway-mutation-invariant, hence blind to what s sees.

Coefficients are exact rationals (fractions.Fraction). Multimodular reduction -- run mod
several primes, then CRT + rational reconstruction -- is a fixed-width *optimization* for
the acceleration phase (decision 0007); the faithful reference is exact Q arithmetic.

Representation mirrors GradedComplex, with one change: a boundary map is a rational
matrix, so a column is the mapping {row index -> coefficient} of its non-zero entries,
not a set of rows. The cochain convention (d^n: C^n -> C^(n+1)) and one-grading-per-
complex rule are unchanged. The front end still splits a multi-graded theory into singly
graded pieces -- but note Lee is only *filtered* by the quantum grading, not graded by
it, so the Lee front end hands over a single complex graded by homological degree alone
(all quantum degrees together), unlike Khovanov which splits by quantum j.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction


def _add_scaled(acc: dict[int, Fraction], scalar: Fraction, col: Mapping[int, Fraction]) -> None:
    """In-place acc += scalar * col over Q, dropping entries that cancel to zero."""
    for r, c in col.items():
        v = acc.get(r, Fraction(0)) + scalar * c
        if v == 0:
            acc.pop(r, None)
        else:
            acc[r] = v


class RationalComplex:
    """A finite cochain complex of Q vector spaces, graded by homological degree."""

    def __init__(
        self,
        dims: Mapping[int, int],
        maps: Mapping[int, Sequence[Mapping[int, object]]],
    ) -> None:
        clean_dims: dict[int, int] = {}
        for n, d in dims.items():
            d = int(d)
            if d < 0:
                raise ValueError(f"C^{n} has negative dimension {d}.")
            if d:
                clean_dims[n] = d
        self._dims: dict[int, int] = clean_dims

        clean_maps: dict[int, tuple[dict[int, Fraction], ...]] = {}
        for n, cols in maps.items():
            cols = list(cols)
            src, tgt = self.dim(n), self.dim(n + 1)
            if len(cols) != src:
                raise ValueError(
                    f"d^{n} has {len(cols)} columns but C^{n} has dimension {src}."
                )
            built: list[dict[int, Fraction]] = []
            for j, col in enumerate(cols):
                column: dict[int, Fraction] = {}
                for r, c in dict(col).items():
                    r, c = int(r), Fraction(c)
                    if c == 0:
                        continue
                    if not 0 <= r < tgt:
                        raise ValueError(
                            f"d^{n} column {j} references row {r}, but C^{n+1} has "
                            f"dimension {tgt}."
                        )
                    column[r] = c
                built.append(column)
            if any(built):  # a zero map carries no information; omit it
                clean_maps[n] = tuple(built)
        self._maps: dict[int, tuple[dict[int, Fraction], ...]] = clean_maps

    def dim(self, n: int) -> int:
        """Dimension of C^n (0 if the degree is absent)."""
        return self._dims.get(n, 0)

    def degrees(self) -> list[int]:
        """Homological degrees with a non-zero chain group, ascending."""
        return sorted(self._dims)

    def differential(self, n: int) -> tuple[dict[int, Fraction], ...]:
        """Columns of d^n: C^n -> C^(n+1). A zero (or absent) map is returned as
        `dim(n)` empty columns, so the length is always `dim(n)`."""
        if n in self._maps:
            return self._maps[n]
        return tuple({} for _ in range(self.dim(n)))

    def total_dim(self) -> int:
        """Sum of all chain-group dimensions: the exact storage of the unreduced
        complex (the cheap-from-diagram predictor is front-end-specific)."""
        return sum(self._dims.values())

    def verify_d_squared(self) -> None:
        """Check d^(n+1) . d^n = 0 over Q at every degree; raise on the first
        violation. A non-zero composite means the front end's differential is wrong --
        exactly the kind of bug to surface loudly (decision 0004)."""
        for n in self._dims:
            if n + 1 not in self._maps:
                continue  # d^(n+1) is the zero map, so the composite is zero
            dn1 = self._maps[n + 1]
            for j, col in enumerate(self.differential(n)):
                acc: dict[int, Fraction] = {}
                for r, c in col.items():     # r is a basis element of C^(n+1)
                    _add_scaled(acc, c, dn1[r])  # add c * column r of d^(n+1)
                if acc:
                    raise RuntimeError(
                        f"d^2 != 0: (d^{n+1} . d^{n}) applied to basis element {j} of "
                        f"C^{n} is non-zero {dict(sorted(acc.items()))} in C^{n+2}."
                    )
