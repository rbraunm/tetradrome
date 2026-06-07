"""Khovanov enhanced states and (i, j) gradings, built on the shared cube skeleton.

An *enhanced state* is a cube state together with a labeling of each of its circles by
+ or -, the two generators of the Frobenius algebra V = F2[x]/(x^2): + is 1 (quantum
degree +1), - is x (quantum degree -1). A state with c circles therefore carries 2^c
enhanced generators.

Grading conventions (standard Khovanov / Bar-Natan; the cube's bit 1 is the
1-smoothing, see engines/cube.py):

    i(s) = |s| - n-                                  (homological)
    j     = (#+  -  #-) + |s| + n+ - 2 n-             (quantum)

where |s| is the number of 1-smoothings (sum of the state bits) and n+, n- are the
positive / negative crossing counts (from the validated Seifert sign computation).
These are pinned the same way the rest of the cube was: the chain-level graded Euler
characteristic, sum_{i,j} (-1)^i q^j dim C^{i,j}, equals the unnormalized Jones
polynomial (q + q^-1) V(q^-2) -- checked in the tests against the KnotInfo-validated
jones.py. (The substitution is q^-2 because jones.py follows KnotInfo's t-variable.)
The Euler characteristic is blind to the sign of i, so it fixes j and i's parity; the
absolute homological orientation is locked against KnotInfo's Khovanov table in the
homology step (Phase 2c).
"""
from __future__ import annotations

import itertools
from collections import Counter

from ..cube import resolve, states
from ...diagrams import seifert_structure
from ...diagrams.model import PDCode


def crossing_counts(pd: PDCode) -> tuple[int, int]:
    """(n_plus, n_minus): positive and negative crossing counts, from the writhe."""
    n = len(pd)
    w = seifert_structure(pd).writhe
    # n+ + n- = n,  n+ - n- = w.
    if (n + w) % 2:
        raise ValueError(f"writhe {w} and crossing count {n} have mismatched parity.")
    n_plus = (n + w) // 2
    return n_plus, n - n_plus


def enhanced_generators(pd: PDCode):
    """Yield (state, circles, labels) for every enhanced generator of the cube.

    `circles` is the tuple of arc-label sets from `resolve` (so the differential can
    later match circles between adjacent states); `labels[k]` is +1 or -1 on circle k.
    """
    for state in states(len(pd)):
        circles = resolve(pd, state)
        for labels in itertools.product((1, -1), repeat=len(circles)):
            yield state, circles, labels


def grading(state, labels, n_plus: int, n_minus: int) -> tuple[int, int]:
    """The (i, j) bidegree of one enhanced generator."""
    s = sum(state)                       # |s|, the number of 1-smoothings
    i = s - n_minus
    j = sum(labels) + s + n_plus - 2 * n_minus   # sum(labels) = (#+) - (#-)
    return i, j


def chain_dimensions(pd: PDCode) -> dict[tuple[int, int], int]:
    """Dimensions of the unreduced Khovanov chain groups: {(i, j): dim C^{i,j}}."""
    n_plus, n_minus = crossing_counts(pd)
    dims: Counter[tuple[int, int]] = Counter()
    for state, _circles, labels in enhanced_generators(pd):
        dims[grading(state, labels, n_plus, n_minus)] += 1
    return dict(dims)


def unreduced_size(pd: PDCode) -> int:
    """Total number of enhanced generators = sum over states of 2^(circles).

    The exact storage of the unreduced complex, read off the diagram without building
    it (homology-engine design section 5). Cube-specific, so it lives with the front
    end, not in the invariant-agnostic algebra layer.
    """
    return sum(1 << len(resolve(pd, state)) for state in states(len(pd)))
