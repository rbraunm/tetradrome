"""The Ozsvath-Szabo tau invariant from grid homology (engine Phase 6).

tau comes from the Alexander *filtration*, not the bigraded associated graded. Take the grid
hat complex with the filtered differential (empty rectangles avoiding the O markers; X's
allowed, so the differential can lower Alexander). Its total homology is that of the ambient
S^3 tensored with the basepoint factor, whose Maslov-grading-0 part is one-dimensional. The
Alexander grading filters the complex by subcomplexes F_i = {states with Alexander <= i}, and

    tau(K) = min{ i : H_0(F_i) -> H_0(total) is nonzero }

(the smallest filtration level carrying a representative of the surviving class). With the grid
in the standard chirality this reproduces KnotInfo's tau; for a knot the surviving class is
one-dimensional, which we check.

Computed over F2 with the shared reducer: from the Maslov 0 / +-1 pieces we take the boundaries
B_0 = im(d: C_1 -> C_0) and, level by level, the cycles supported on Alexander <= i (a kernel);
tau is the first level whose cycles are not all boundaries (an augmented rank exceeds rank d_1).
"""
from __future__ import annotations

from collections import defaultdict

from ...algebra import f2_kernel, f2_rank
from ...errors import UnvalidatedResult
from .differential import filtered_differential
from .gradings import alexander, maslov


def tau(grid) -> int:
    """The tau invariant of the knot presented by ``grid`` (standard chirality)."""
    by_maslov: dict = defaultdict(list)
    for state in grid.generators():
        by_maslov[maslov(grid, state)].append(state)
    c0, c_minus, c1 = by_maslov[0], by_maslov[-1], by_maslov[1]

    minus_index = {state: i for i, state in enumerate(c_minus)}
    zero_index = {state: i for i, state in enumerate(c0)}

    # boundary maps as columns = sets of row indices
    d0 = [
        frozenset(minus_index[y] for y in filtered_differential(grid, s) if y in minus_index)
        for s in c0
    ]
    d1 = [
        frozenset(zero_index[y] for y in filtered_differential(grid, s) if y in zero_index)
        for s in c1
    ]
    rank_d1 = f2_rank(d1)

    h0_dim = len(c0) - f2_rank(d0) - rank_d1
    if h0_dim != 1:
        raise UnvalidatedResult(
            f"filtered grid homology has dim H_0 = {h0_dim}, expected 1 for a knot."
        )

    alexander_of = [alexander(grid, s) for s in c0]
    for level in sorted(set(alexander_of)):
        below = [k for k in range(len(c0)) if alexander_of[k] <= level]
        cycles = [
            frozenset(below[p] for p in relation)            # lift column indices back to C_0
            for relation in f2_kernel([d0[k] for k in below])
        ]
        if f2_rank(d1 + cycles) > rank_d1:
            return level
    raise UnvalidatedResult("no Alexander level carried the surviving class; tau undefined.")
