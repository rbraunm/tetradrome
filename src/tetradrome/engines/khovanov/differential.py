"""The Khovanov differential, assembled into one complex per quantum grading.

Each edge of the cube flips a single crossing 0 -> 1, which (compared as partitions of
the arc labels) either merges two circles into one or splits one into two. The induced
map is the Frobenius algebra V = F[x]/(x^2) acting on the involved circles and the
identity elsewhere:

    m :  + (x) + -> + ,   + (x) - -> - ,   - (x) + -> - ,   - (x) - -> 0
    D :  +      -> + (x) - + - (x) + ,      -      -> - (x) -

(+ is the unit 1 of quantum degree +1, - is x of quantum degree -1.)

Two lanes share this construction. Over F2 the cube edge signs vanish, so the
differential is the plain F2 sum of these maps -- the fast lane, `khovanov_complexes`
(decision 0003). Over Q the edges carry the standard sign assignment

    sign(flip crossing c at state s) = (-1)^(number of 1-bits of s before position c),

which makes every cube square anticommute (the unsigned squares commute by the Frobenius
relations), hence d^2 = 0 over Z -- the rational lane, `khovanov_complexes_q`, needed by
Lee / Rasmussen. The two agree mod 2 by construction.

The differential preserves the quantum grading j and raises the homological grading i by
1, so the complex splits as a direct sum over j of cochain complexes in i. Both lanes
perform that split and return one complex per j for the back end to reduce;
verify_d_squared() on each is the correctness check on the whole construction (and, over
Q, on the sign assignment specifically).
"""
from __future__ import annotations

import itertools

from ...algebra import GradedComplex, RationalComplex
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


def _edge_sign(state, c: int) -> int:
    """Standard cube sign for flipping crossing c at `state`: (-1)^(1-bits before c)."""
    return -1 if (sum(state[:c]) & 1) else 1


def _edge_targets(state, labeling, resolved):
    """Apply the differential along every 0 -> 1 edge from `state`.

    `labeling` maps each circle of `state` (a frozenset of arcs) to its sign; `resolved`
    maps a state to its tuple of circles. Yields (crossing, state', labeling') for each
    target generator; the crossing is the flipped coordinate, so the caller can sign it.
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
            yield c, s2, {**carried, merged: label}
        elif len(B) == len(A) + 1:               # split: one circle -> two
            (split,) = tuple(a_only)
            c1, c2 = tuple(b_only)
            for l1, l2 in _comultiply(labeling[split]):
                yield c, s2, {**carried, c1: l1, c2: l2}
        else:
            raise RuntimeError(
                f"edge at crossing {c} changed the circle count by {len(B) - len(A)} "
                f"(a single smoothing change must merge or split, i.e. +/-1)."
            )


def _raw_differential(pd: PDCode):
    """Shared scaffolding for both lanes: enumerate and index the enhanced generators,
    then for each generator collect its differential targets as (target_index, crossing)
    pairs. The crossing lets the rational lane sign each contribution; the F2 lane
    ignores it. Returns (by_grading, raw), where by_grading[(i,j)] lists the generator
    keys in index order and raw[key] is the list of (target_index, crossing) pairs.
    """
    if not pd:
        raise ValueError(
            "Khovanov cube on an empty diagram. The crossingless unknot is handled at "
            "the invariant level, not in the cube."
        )
    n_plus, n_minus = crossing_counts(pd)
    resolved = {state: resolve(pd, state) for state in states(len(pd))}

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

    raw: dict[tuple, list[tuple[int, int]]] = {}
    for (i, j), keys in by_grading.items():
        for key in keys:
            state, _ = key
            pairs: list[tuple[int, int]] = []
            for c, s2, lab2 in _edge_targets(state, labeling_of[key], resolved):
                i2, j2, idx2 = index[(s2, frozenset(lab2.items()))]
                if i2 != i + 1 or j2 != j:
                    raise RuntimeError(
                        f"differential left its grading: ({i},{j}) -> ({i2},{j2})."
                    )
                pairs.append((idx2, c))
            raw[key] = pairs
    return by_grading, raw


def _quantum_gradings(by_grading) -> list[int]:
    return sorted({j for (_i, j) in by_grading})


def khovanov_complexes(pd: PDCode) -> dict[int, GradedComplex]:
    """The Khovanov cochain complex over F2 (the fast lane), one GradedComplex per
    quantum grading. Edge signs vanish mod 2, so a column is the parity of its targets.
    """
    by_grading, raw = _raw_differential(pd)
    complexes: dict[int, GradedComplex] = {}
    for j in _quantum_gradings(by_grading):
        i_values = sorted(i for (i, jj) in by_grading if jj == j)
        dims = {i: len(by_grading[(i, j)]) for i in i_values}
        maps: dict[int, list[frozenset[int]]] = {}
        for i in i_values:
            columns: list[frozenset[int]] = []
            for key in by_grading[(i, j)]:
                acc: set[int] = set()
                for idx2, _c in raw[key]:
                    acc ^= {idx2}                # F2: equal targets cancel in pairs
                columns.append(frozenset(acc))
            maps[i] = columns
        complexes[j] = GradedComplex(dims, maps)
    return complexes


def khovanov_complexes_q(pd: PDCode) -> dict[int, RationalComplex]:
    """The Khovanov cochain complex over Q (the rational lane), one RationalComplex per
    quantum grading. Each target is weighted by the standard cube edge sign, so a column
    is the signed sum of its targets; d^2 = 0 over Q certifies the sign assignment.
    """
    by_grading, raw = _raw_differential(pd)
    complexes: dict[int, RationalComplex] = {}
    for j in _quantum_gradings(by_grading):
        i_values = sorted(i for (i, jj) in by_grading if jj == j)
        dims = {i: len(by_grading[(i, j)]) for i in i_values}
        maps: dict[int, list[dict[int, int]]] = {}
        for i in i_values:
            columns: list[dict[int, int]] = []
            for key in by_grading[(i, j)]:
                state, _ = key
                col: dict[int, int] = {}
                for idx2, c in raw[key]:
                    col[idx2] = col.get(idx2, 0) + _edge_sign(state, c)
                columns.append({r: v for r, v in col.items() if v})
            maps[i] = columns
        complexes[j] = RationalComplex(dims, maps)
    return complexes
