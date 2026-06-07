"""The grid (knot Floer) differentials: empty rectangles between grid states (Phase 6).

Two grid states connected by transposing the points in two rows bound a rectangle. The
differential counts the *empty* rectangles (containing no other state point) that also avoid
a prescribed set of markers; enumerating ordered row pairs (i, j) walks both toroidal
rectangles of each transposition (south-west corner on the state point in row i, extending up
to row j and right to column x[j]). Every rectangle lowers the Maslov grading by one.

Two flavours share the enumeration:

* ``differential`` avoids *all* markers (O and X). It preserves the Alexander grading, so it
  makes each Alexander grading a Maslov-graded F2 complex -- the associated graded, whose
  homology is HFK-hat (x) V^{n-1}.
* ``filtered_differential`` avoids only the O markers (X's allowed). An X inside a rectangle
  lowers the Alexander grading, so this differential is filtered, not graded, by Alexander;
  its homology is that of the ambient S^3 and the Alexander filtration carries tau.

Both square to zero.
"""
from __future__ import annotations

from collections import defaultdict


def _empty_rectangles(grid, x, avoid):
    """Yield the target state of every empty rectangle out of ``x`` that avoids the cells in
    ``avoid`` (a set of (row, column) pairs)."""
    n = grid.n
    for i in range(n):
        ci = x[i]
        for j in range(n):
            if i == j:
                continue
            height = (j - i) % n
            width = (x[j] - ci) % n
            open_rows = {(i + t) % n for t in range(1, height)}
            open_cols = {(ci + t) % n for t in range(1, width)}
            if any(x[k] in open_cols for k in open_rows):
                continue                                  # another state point inside
            cell_rows = {(i + t) % n for t in range(height)}
            cell_cols = {(ci + t) % n for t in range(width)}
            if any(mr in cell_rows and mc in cell_cols for mr, mc in avoid):
                continue                                  # a forbidden marker inside
            y = list(x)
            y[i], y[j] = x[j], ci
            yield tuple(y)


def _reduce_mod_2(targets):
    counts: dict = defaultdict(int)
    for y in targets:
        counts[y] += 1
    return {y: 1 for y, count in counts.items() if count % 2}


def differential(grid, sigma) -> dict:
    """Bigraded (hat) differential ``d(sigma)`` as ``{target: 1}`` over F2: empty rectangles
    avoiding every O and X marker."""
    return _reduce_mod_2(_empty_rectangles(grid, list(sigma), grid.marker_cells))


def filtered_differential(grid, sigma) -> dict:
    """Alexander-filtered differential ``d(sigma)`` as ``{target: 1}`` over F2: empty
    rectangles avoiding the O markers only."""
    return _reduce_mod_2(_empty_rectangles(grid, list(sigma), grid.o_cells))
