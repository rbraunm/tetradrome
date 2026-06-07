"""The Khovanov differential over F2, assembled into one complex per quantum grading.

Each edge of the cube flips a single crossing 0 -> 1, which (compared as partitions of
the arc labels) either merges two circles into one or splits one into two. The induced
map is the Frobenius algebra V = F2[x]/(x^2) acting on the involved circles and the
identity elsewhere:

    m :  + (x) + -> + ,   + (x) - -> - ,   - (x) + -> - ,   - (x) - -> 0
    D :  +      -> + (x) - + - (x) + ,      -      -> - (x) -

(+ is the unit 1 of quantum degree +1, - is x of quantum degree -1.) Over F2 the edge
signs vanish, so the differential is just the F2 sum of these maps -- which is why
decision 0003 puts F2 first.

The differential preserves the quantum grading j and raises the homological grading i
by 1, so the complex splits as a direct sum over j of cochain complexes in i. This
module performs that split and returns one `GradedComplex` per j for the shared back
end to reduce; `verify_d_squared()` on each is the correctness check on the whole
construction.
"""
from __future__ import annotations

import itertools

from ...algebra import GradedComplex
from ...diagrams.model import PDCode
from ..cube import resolve, states
from .gradings import crossing_counts


def _multiply(s1: int, s2: int) -> int | None:
    """Frobenius multiplication m(s1, s2); None is the zero element (x . x)."""
    if s1 == 1 and s2 == 1:
        return 1            # 1 . 1 = 1
    if s1 == -1 and s2 == -1:
        return None         # x . x = 0
    return -1               # 1 . x = x . 1 = x


def _comultiply(s: int) -> tuple[tuple[int, int], ...]:
    """Frobenius comultiplication D(s), as the terms (label1, label2)."""
    if s == 1:
        return ((1, -1), (-1, 1))   # 1 -> 1(x)x + x(x)1
    return ((-1, -1),)              # x -> x(x)x


def _edge_targets(state, labeling, resolved):
    """Apply the differential along every 0 -> 1 edge from `state`.

    `labeling` maps each circle of `state` (a frozenset of arcs) to its sign; `resolved`
    maps a state to its tuple of circles. Yields (state', labeling') target generators.
    """
    n = len(state)
    A = set(resolved[state])
    for c in range(n):
        if state[c] == 1:
            continue
        s2 = state[:c] + (1,) + state[c + 1:]
        B = set(resolved[s2])
        spectators = A & B
        a_only, b_only = A - B, B - A
        carried = {circ: labeling[circ] for circ in spectators}
        if len(B) == len(A) - 1:                 # merge: two circles -> one
            c1, c2 = tuple(a_only)
            (merged,) = tuple(b_only)
            label = _multiply(labeling[c1], labeling[c2])
            if label is None:                    # x . x = 0, edge contributes nothing
                continue
            yield s2, {**carried, merged: label}
        elif len(B) == len(A) + 1:               # split: one circle -> two
            (split,) = tuple(a_only)
            c1, c2 = tuple(b_only)
            for l1, l2 in _comultiply(labeling[split]):
                yield s2, {**carried, c1: l1, c2: l2}
        else:
            raise RuntimeError(
                f"edge at crossing {c} changed the circle count by {len(B) - len(A)} "
                f"(a single smoothing change must merge or split, i.e. +/-1)."
            )


def khovanov_complexes(pd: PDCode) -> dict[int, GradedComplex]:
    """The Khovanov cochain complex over F2, one `GradedComplex` per quantum grading.

    Raises on the empty diagram: the crossingless unknot is the cube's representational
    boundary (0 circles), handled at the invariant level (Phase 2c), not here -- the
    same split kauffman_bracket makes.
    """
    if not pd:
        raise ValueError(
            "khovanov_complexes: empty diagram. The crossingless unknot is handled at "
            "the invariant level, not in the cube."
        )
    n_plus, n_minus = crossing_counts(pd)
    resolved = {state: resolve(pd, state) for state in states(len(pd))}

    # Index every generator within its (i, j) class so the differential can be written
    # column-wise. index[key] = (i, j, position); by_grading[(i,j)] = keys in order.
    by_grading: dict[tuple[int, int], list] = {}
    index: dict[tuple, tuple[int, int, int]] = {}
    labeling_of: dict[tuple, dict] = {}

    for state, circles in resolved.items():
        s = sum(state)
        i = s - n_minus
        for bits in itertools.product((1, -1), repeat=len(circles)):
            labeling = dict(zip(circles, bits))
            j = sum(bits) + s + n_plus - 2 * n_minus
            key = (state, frozenset(labeling.items()))
            bucket = by_grading.setdefault((i, j), [])
            index[key] = (i, j, len(bucket))
            bucket.append(key)
            labeling_of[key] = labeling

    quantum = sorted({j for (_i, j) in by_grading})
    complexes: dict[int, GradedComplex] = {}
    for j in quantum:
        i_values = sorted(i for (i, jj) in by_grading if jj == j)
        dims = {i: len(by_grading[(i, j)]) for i in i_values}
        maps: dict[int, list[frozenset[int]]] = {}
        for i in i_values:
            columns: list[frozenset[int]] = []
            for key in by_grading[(i, j)]:
                state, _ = key
                targets: set[int] = set()
                for s2, lab2 in _edge_targets(state, labeling_of[key], resolved):
                    i2, j2, idx2 = index[(s2, frozenset(lab2.items()))]
                    if i2 != i + 1 or j2 != j:
                        raise RuntimeError(
                            f"differential left its grading: ({i},{j}) -> ({i2},{j2})."
                        )
                    targets ^= {idx2}          # F2: equal targets cancel in pairs
                columns.append(frozenset(targets))
            maps[i] = columns
        complexes[j] = GradedComplex(dims, maps)
    return complexes
