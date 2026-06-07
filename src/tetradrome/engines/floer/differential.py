"""The grid (knot Floer) differential: empty rectangles avoiding all markers (Phase 6).

Two grid states connected by transposing the points in two rows bound a rectangle. For the
hat flavour over F2 the differential counts the *empty* rectangles (containing no other
state point) that also avoid every marker -- no O and no X inside:

    d(x) = sum_y #{ empty marker-avoiding rectangles x -> y } * y    (mod 2).

Enumerating ordered row pairs (i, j) walks both toroidal rectangles of each transposition
(south-west corner on the state point in row i, extending up to row j and right to column
x[j]); a rectangle is dropped if any other state point lies in its open interior or if any
marker lies in one of its cells. The differential lowers the Maslov grading by one, preserves
the Alexander grading, and squares to zero.
"""
from __future__ import annotations

from collections import defaultdict


def differential(grid, sigma) -> dict:
    """``d(sigma)`` as ``{target permutation: 1}`` over F2."""
    n = grid.n
    x = list(sigma)
    markers = grid.marker_cells
    out: dict = defaultdict(int)
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
            if any(mr in cell_rows and mc in cell_cols for mr, mc in markers):
                continue                                  # an O or X marker inside
            y = list(x)
            y[i], y[j] = x[j], ci
            out[tuple(y)] += 1
    return {y: 1 for y, count in out.items() if count % 2}
