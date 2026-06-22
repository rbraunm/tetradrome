# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Lee deformation of Khovanov homology over Q.

Lee keeps the cube, its enhanced-state generators, and the cube edge signs, but deforms
the Frobenius algebra to V = Q[x]/(x^2 - 1):

    m_Lee:  + . + -> + ,  + . - -> - ,  - . + -> - ,  - . - -> +    (x.x = 1, not 0)
    D_Lee:  + -> +(x)- + -(x)+ ,        - -> -(x)- + +(x)+          (x -> x(x)x + 1(x)1)

With the same signed cube, d_Lee^2 = 0 over Q. The deformation raises the quantum degree
by 4, so the Lee differential does NOT preserve j: unlike Khovanov it does not split by
quantum grading. The complex is filtered by q and graded by homological degree i alone,
so the front end hands the back end a single complex graded by i.

Lee's theorem: the Lee homology of an n-component link is 2^n-dimensional over Q -- for a
knot, exactly 2. That dimension is the validation here. The quantum filtration on those
two surviving generators is what pins Rasmussen's s, read off in a later step.
"""
from __future__ import annotations

from ...algebra import RationalComplex, rational_homology
from ...diagrams.model import PDCode
from .differential import _edge_sign, _edge_targets, _enumerate_generators


def _multiply_lee(s1: int, s2: int) -> int:
    """Lee multiplication: x.x = 1 (the deformation), never zero."""
    return 1 if s1 == s2 else -1     # +.+ = -.- = + ;  +.- = -.+ = -


def _comultiply_lee(s: int) -> tuple[tuple[int, int], ...]:
    """Lee comultiplication; D(x) gains the 1(x)1 term over Khovanov."""
    if s == 1:
        return ((1, -1), (-1, 1))        # + -> +(x)- + -(x)+
    return ((-1, -1), (1, 1))            # - -> -(x)- + +(x)+


def _assemble_lee(pd: PDCode):
    """Build the Lee complex data graded by homological degree i. Returns
    (dims, maps, qdeg), where qdeg[i][pos] is the quantum degree of the pos-th generator
    of C^i, in the same basis order as the maps. The q-grading is a filtration (not a
    grading) of the Lee differential; the s-invariant reads it off this basis."""
    resolved, gens = _enumerate_generators(pd)   # raises on the empty diagram
    by_i: dict[int, list] = {}
    index: dict[tuple, tuple[int, int]] = {}      # key -> (i, position)
    labeling_of: dict[tuple, dict] = {}
    qdeg: dict[int, list[int]] = {}
    for i, j, _state, labeling, key in gens:
        bucket = by_i.setdefault(i, [])
        index[key] = (i, len(bucket))
        bucket.append(key)
        labeling_of[key] = labeling
        qdeg.setdefault(i, []).append(j)

    dims = {i: len(keys) for i, keys in by_i.items()}
    maps: dict[int, list[dict[int, int]]] = {}
    for i, keys in by_i.items():
        columns: list[dict[int, int]] = []
        for key in keys:
            state, _ = key
            col: dict[int, int] = {}
            for c, s2, lab2 in _edge_targets(
                state, labeling_of[key], resolved,
                multiply=_multiply_lee, comultiply=_comultiply_lee,
            ):
                i2, pos2 = index[(s2, frozenset(lab2.items()))]
                if i2 != i + 1:
                    raise RuntimeError(f"Lee differential left its grading: {i} -> {i2}.")
                col[pos2] = col.get(pos2, 0) + _edge_sign(state, c)
            columns.append({r: v for r, v in col.items() if v})
        maps[i] = columns
    return dims, maps, qdeg


def lee_complex(pd: PDCode) -> RationalComplex:
    """The Lee cochain complex over Q: a single complex graded by homological degree i
    (the deformation breaks the quantum grading, so there is no split by j)."""
    dims, maps, _qdeg = _assemble_lee(pd)
    return RationalComplex(dims, maps)


def lee_complex_graded(pd: PDCode) -> tuple[RationalComplex, dict[int, list[int]]]:
    """The Lee complex together with qdeg[i][pos], the quantum degree of each generator,
    for reading off the quantum filtration (Rasmussen's s)."""
    dims, maps, qdeg = _assemble_lee(pd)
    return RationalComplex(dims, maps), qdeg


def lee_homology(pd: PDCode) -> dict[int, int]:
    """Lee homology over Q as {i: dim}. The crossingless unknot is the cube boundary,
    handled here like the other invariants: 2-dimensional in homological degree 0."""
    if not pd:
        return {0: 2}
    return rational_homology(lee_complex(pd))   # verifies d^2 = 0 over Q
