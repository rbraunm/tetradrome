"""Graded chain complexes over F2 -- the shared back-end data structure.

This is the invariant-agnostic core (SPEC.md 13.6; homology-engine design section 2):
a front end (Khovanov, later Floer) emits one of these per independent grading, and
the back end reduces it to homology. F2 first (decision 0003): a boundary map is a 0/1
matrix, and this is the bit-set fast lane for it. The rational coefficients Lee /
Rasmussen need live beside this as RationalComplex (a coefficient-column representation
reduced over Q), not as a generalization of this class -- the field-tested pattern keeps
an F2 fast path next to a general-field one (Ripser, PHAT) rather than forcing one
representation to serve both.

Conventions
-----------
- Cochain convention: the differential RAISES homological degree, d^n: C^n -> C^(n+1).
  This matches Khovanov, the first consumer. (A chain complex with a lowering
  differential is the same object reindexed; the front end picks the indexing.)
- Single grading per complex. Khovanov is bigraded (homological i, quantum j), but the
  differential preserves j, so the complex splits as a direct sum over j of independent
  cochain complexes in i. The front end performs that split and hands the back end one
  singly-graded complex at a time; the back end stays grading-agnostic. (Floer splits
  the same way by its Alexander grading.)
- F2 coefficients: a boundary map is a 0/1 matrix, stored column-wise as the set of row
  indices whose entry is 1. Vector addition is symmetric difference (XOR).

Representation
--------------
- `dims[n]`  = dimension (number of basis elements) of the chain group C^n. A degree
  absent from `dims` (or mapped to 0) has dimension 0.
- `maps[n]`  = the matrix of d^n: C^n -> C^(n+1), given column-wise. `maps[n][j]` is the
  set of row indices (basis elements of C^(n+1)) appearing in d^n(e_j), where e_j is the
  j-th basis element of C^n. So `len(maps[n]) == dims[n]`, and every index in a column
  lies in `range(dims[n+1])`.

Construction validates this structure fully and fails loud on any inconsistency: a
column whose length or row indices disagree with the declared dimensions is a bug in
the front end that built it, not something to coerce into shape.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def _xor_into(acc: set[int], col: Iterable[int]) -> None:
    """In-place F2 vector addition: acc ^= col (symmetric difference of supports)."""
    acc.symmetric_difference_update(col)


class GradedComplex:
    """A finite cochain complex of F2 vector spaces, graded by homological degree."""

    def __init__(
        self,
        dims: Mapping[int, int],
        maps: Mapping[int, Sequence[Iterable[int]]],
    ) -> None:
        clean_dims: dict[int, int] = {}
        for n, d in dims.items():
            d = int(d)
            if d < 0:
                raise ValueError(f"C^{n} has negative dimension {d}.")
            if d:
                clean_dims[n] = d
        self._dims: dict[int, int] = clean_dims

        clean_maps: dict[int, tuple[frozenset[int], ...]] = {}
        for n, cols in maps.items():
            cols = list(cols)
            src, tgt = self.dim(n), self.dim(n + 1)
            if len(cols) != src:
                raise ValueError(
                    f"d^{n} has {len(cols)} columns but C^{n} has dimension {src}."
                )
            built: list[frozenset[int]] = []
            for j, col in enumerate(cols):
                rows = frozenset(int(r) for r in col)
                for r in rows:
                    if not 0 <= r < tgt:
                        raise ValueError(
                            f"d^{n} column {j} references row {r}, but C^{n+1} has "
                            f"dimension {tgt}."
                        )
                built.append(rows)
            if any(built):  # a zero map carries no information; omit it
                clean_maps[n] = tuple(built)
        self._maps: dict[int, tuple[frozenset[int], ...]] = clean_maps

    def dim(self, n: int) -> int:
        """Dimension of C^n (0 if the degree is absent)."""
        return self._dims.get(n, 0)

    def degrees(self) -> list[int]:
        """Homological degrees with a non-zero chain group, ascending."""
        return sorted(self._dims)

    def differential(self, n: int) -> tuple[frozenset[int], ...]:
        """Columns of d^n: C^n -> C^(n+1). A zero (or absent) map is returned as
        `dim(n)` empty columns, so the length is always `dim(n)`."""
        if n in self._maps:
            return self._maps[n]
        return tuple(frozenset() for _ in range(self.dim(n)))

    def total_dim(self) -> int:
        """Sum of all chain-group dimensions: the exact storage of the *unreduced*
        complex (homology-engine design section 5). The cheaper predictor that reads
        the size off the diagram without building the complex is front-end-specific
        and lands with Khovanov in Phase 2."""
        return sum(self._dims.values())

    def verify_d_squared(self) -> None:
        """Check d^(n+1) . d^n = 0 over F2 at every degree; raise on the first
        violation. A non-zero composite means the front end's differential is wrong,
        which is exactly the kind of bug to surface loudly (decision 0004, inward)."""
        for n in self._dims:
            if n + 1 not in self._maps:
                continue  # d^(n+1) is the zero map, so the composite is zero
            dn1 = self._maps[n + 1]
            for j, col in enumerate(self.differential(n)):
                acc: set[int] = set()
                for r in col:            # r is a basis element of C^(n+1)
                    _xor_into(acc, dn1[r])  # add column r of d^(n+1)
                if acc:
                    raise RuntimeError(
                        f"d^2 != 0: (d^{n+1} . d^{n}) applied to basis element {j} of "
                        f"C^{n} is non-zero {sorted(acc)} in C^{n+2}."
                    )
