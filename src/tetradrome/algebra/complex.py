# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

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
- F2 coefficients: a boundary map is a 0/1 matrix, stored column-wise as the row indices
  whose entry is 1. Vector addition is symmetric difference (XOR).

Representation
--------------
- `dims[n]` = dimension (number of basis elements) of the chain group C^n. A degree
  absent from `dims` (or mapped to 0) has dimension 0.
- The matrix of d^n: C^n -> C^(n+1) is stored column-major in compressed-sparse-column
  form over two `array('i')` (4-byte int) buffers: `indices[n]` is the row indices of
  every column concatenated, and `indptr[n]` (length `dim(n)+1`) is each column's start,
  so column j of d^n is `indices[n][indptr[n][j]:indptr[n][j+1]]` -- the basis elements of
  C^(n+1) appearing in d^n(e_j) -- with every index in `range(dim(n+1))`. A zero map is
  omitted.
- CSC of two array buffers, not per-column Python objects, because a reduce worker
  receives its complex as a pickled copy through the scheduler: the buffers pickle and
  unpickle ~1000x faster and ~25x smaller than the equivalent frozensets, and the F2
  reducers read them directly -- the numpy tiers via a zero-copy `np.frombuffer`, the
  pure-Python tiers by slicing. `array` is stdlib, so this stays numpy-free; numpy lives
  only in the acceleration tiers that already require it.

Construction validates this structure fully and fails loud on any inconsistency: a
column whose length or row indices disagree with the declared dimensions is a bug in the
front end that built it, not something to coerce into shape. The readable constructor
takes the matrix column-by-column (any iterable of row-index iterables) and packs it;
`from_csc` takes the CSC buffers directly -- the fast path for a producer that already
has them (the Floer merge).
"""
from __future__ import annotations

from array import array
from collections.abc import Iterable, Mapping, Sequence

_INDEX_TYPECODE = "i"      # 4-byte signed int: row indices and column offsets


def _clean_dims(dims: Mapping[int, int]) -> dict[int, int]:
    clean: dict[int, int] = {}
    for n, d in dims.items():
        d = int(d)
        if d < 0:
            raise ValueError(f"C^{n} has negative dimension {d}.")
        if d:
            clean[n] = d
    return clean


class GradedComplex:
    """A finite cochain complex of F2 vector spaces, graded by homological degree."""

    def __init__(
        self,
        dims: Mapping[int, int],
        maps: Mapping[int, Sequence[Iterable[int]]],
    ) -> None:
        self._dims = _clean_dims(dims)
        self._indices: dict[int, array] = {}
        self._indptr: dict[int, array] = {}
        for n, cols in maps.items():
            cols = list(cols)
            src, tgt = self.dim(n), self.dim(n + 1)
            if len(cols) != src:
                raise ValueError(
                    f"d^{n} has {len(cols)} columns but C^{n} has dimension {src}."
                )
            indices = array(_INDEX_TYPECODE)
            indptr = array(_INDEX_TYPECODE, [0])
            for j, col in enumerate(cols):
                rows: set[int] = set()
                for r in col:
                    r = int(r)
                    if not 0 <= r < tgt:
                        raise ValueError(
                            f"d^{n} column {j} references row {r}, but C^{n+1} has "
                            f"dimension {tgt}."
                        )
                    rows.add(r)
                indices.extend(sorted(rows))
                indptr.append(len(indices))
            if len(indices):  # a zero map carries no information; omit it
                self._indices[n] = indices
                self._indptr[n] = indptr

    @classmethod
    def from_csc(
        cls,
        dims: Mapping[int, int],
        csc: Mapping[int, tuple[Iterable[int], Iterable[int]]],
    ) -> "GradedComplex":
        """Build from per-degree CSC buffers ``{n: (indices, indptr)}`` directly, bypassing
        per-column construction -- the fast path for a producer that already holds CSC (the
        Floer merge). Each ``indices``/``indptr`` is an ``array('i')`` (or anything ``array``
        accepts). Validates the CSC structure -- column count and monotone offsets -- and
        trusts the row values; the front end that built the positions owns their range."""
        self = cls.__new__(cls)
        self._dims = _clean_dims(dims)
        self._indices = {}
        self._indptr = {}
        for n, (indices, indptr) in csc.items():
            indices = indices if isinstance(indices, array) else array(_INDEX_TYPECODE, indices)
            indptr = indptr if isinstance(indptr, array) else array(_INDEX_TYPECODE, indptr)
            src, tgt = self.dim(n), self.dim(n + 1)
            if len(indptr) != src + 1:
                raise ValueError(
                    f"d^{n}: indptr length {len(indptr)} != dim(C^{n}) + 1 = {src + 1}."
                )
            if indptr[0] != 0 or indptr[-1] != len(indices):
                raise ValueError(
                    f"d^{n}: indptr ends [{indptr[0]}, {indptr[-1]}] inconsistent with "
                    f"{len(indices)} stored entries."
                )
            for j in range(src):
                if indptr[j + 1] < indptr[j]:
                    raise ValueError(f"d^{n}: indptr decreases at column {j}.")
            if tgt == 0 and len(indices):
                raise ValueError(f"d^{n} has entries but C^{n+1} has dimension 0.")
            if len(indices):
                self._indices[n] = indices
                self._indptr[n] = indptr
        return self

    def dim(self, n: int) -> int:
        """Dimension of C^n (0 if the degree is absent)."""
        return self._dims.get(n, 0)

    def degrees(self) -> list[int]:
        """Homological degrees with a non-zero chain group, ascending."""
        return sorted(self._dims)

    def differential(self, n: int) -> tuple[array, array]:
        """Columns of d^n: C^n -> C^(n+1) in CSC form, as ``(indices, indptr)`` over
        ``array('i')``: column j is ``indices[indptr[j]:indptr[j+1]]`` (row indices in
        C^(n+1)), and ``len(indptr) - 1 == dim(n)``. A zero or absent map comes back as
        empty ``indices`` with an all-zero ``indptr`` of length ``dim(n) + 1``."""
        if n in self._indices:
            return self._indices[n], self._indptr[n]
        return array(_INDEX_TYPECODE), array(_INDEX_TYPECODE, [0]) * (self.dim(n) + 1)

    def nnz(self, n: int) -> int:
        """Stored nonzeros in d^n: the count of 1 entries (basis-element/column incidences).
        Zero for an absent or zero map."""
        return len(self._indices[n]) if n in self._indices else 0

    def total_dim(self) -> int:
        """Sum of all chain-group dimensions: the exact storage of the *unreduced*
        complex (homology-engine design section 5)."""
        return sum(self._dims.values())

    def verify_d_squared(self) -> None:
        """Check d^(n+1) . d^n = 0 over F2 at every degree; raise on the first violation.
        A non-zero composite means the front end's differential is wrong, which is exactly
        the kind of bug to surface loudly (decision 0004, inward)."""
        for n in self._dims:
            if n not in self._indices or n + 1 not in self._indices:
                continue  # d^n or d^(n+1) is the zero map, so the composite is zero
            idx0, ptr0 = self._indices[n], self._indptr[n]
            idx1, ptr1 = self._indices[n + 1], self._indptr[n + 1]
            for j in range(len(ptr0) - 1):
                acc: set[int] = set()
                for r in idx0[ptr0[j]:ptr0[j + 1]]:      # r is a basis element of C^(n+1)
                    acc.symmetric_difference_update(idx1[ptr1[r]:ptr1[r + 1]])  # add its column
                if acc:
                    raise RuntimeError(
                        f"d^2 != 0: (d^{n+1} . d^{n}) applied to basis element {j} of "
                        f"C^{n} is non-zero {sorted(acc)} in C^{n+2}."
                    )
