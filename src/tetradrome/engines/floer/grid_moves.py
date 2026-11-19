# SPDX-License-Identifier: Apache-2.0

"""Grid moves that keep the knot, and a search that uses them to make a grid smaller.

The grid is read on the torus, rows and columns both cyclic, so a cyclic shift of either keeps
the knot and every rule below is stated cyclically.

- Commutation: two neighbouring columns trade places when the rows of one column's two markers
  do not separate the rows of the other's on the circle of rows (and the four rows differ).
  Likewise for neighbouring rows.
- Destabilization: a row whose two markers sit in neighbouring columns is a horizontal segment
  of length one that crosses nothing. Every vertical segment passes over every horizontal one,
  so that segment and the two vertical segments at its ends can be laid together in the plane
  above all horizontal segments and straightened into one vertical segment: the row goes and
  the two columns merge. Likewise for a column whose markers sit in neighbouring rows, in the
  plane below.

Every move gives a grid of the same knot, so the search changes how large the grid is, never
what is computed from it. It removes rows while one can be removed directly, and otherwise
looks for a short sequence of commutations after which one can.
"""
from __future__ import annotations

from collections import deque

from .grid import GridDiagram

Markers = tuple  # (O, X): O[row] and X[row] are the columns of the row's two markers


def _transposed(markers: Markers) -> Markers:
    o_markers, x_markers = markers
    size = len(o_markers)
    o_rows, x_rows = [0] * size, [0] * size
    for row in range(size):
        o_rows[o_markers[row]] = row
        x_rows[x_markers[row]] = row
    return tuple(o_rows), tuple(x_rows)


def _row_destabilized(markers: Markers) -> Markers | None:
    o_markers, x_markers = markers
    size = len(o_markers)
    if size <= 2:
        return None
    for row in range(size):
        gap = (x_markers[row] - o_markers[row]) % size
        if gap == 1:
            left = o_markers[row]
        elif gap == size - 1:
            left = x_markers[row]
        else:
            continue

        def merged(column):
            shifted = (column - left) % size
            return 0 if shifted <= 1 else shifted - 1

        return (tuple(merged(o_markers[r]) for r in range(size) if r != row),
                tuple(merged(x_markers[r]) for r in range(size) if r != row))
    return None


def destabilized(markers: Markers) -> Markers | None:
    """The grid one smaller by a destabilization, or None when no row or column allows one."""
    smaller = _row_destabilized(markers)
    if smaller is not None:
        return smaller
    smaller = _row_destabilized(_transposed(markers))
    return None if smaller is None else _transposed(smaller)


def _separate(first: tuple, second: tuple, size: int) -> bool:
    """Whether the pair ``second`` has exactly one point strictly inside the cyclic interval
    between the points of ``first``."""
    start, length = first[0], (first[1] - first[0]) % size

    def inside(point):
        return 0 < (point - start) % size < length

    return inside(second[0]) != inside(second[1])


def _row_commutations(markers: Markers):
    o_markers, x_markers = markers
    size = len(o_markers)
    for row in range(size):
        below = (row + 1) % size
        first = (o_markers[row], x_markers[row])
        second = (o_markers[below], x_markers[below])
        if len(set(first + second)) < 4 or _separate(first, second, size):
            continue
        o_swapped, x_swapped = list(o_markers), list(x_markers)
        o_swapped[row], o_swapped[below] = o_swapped[below], o_swapped[row]
        x_swapped[row], x_swapped[below] = x_swapped[below], x_swapped[row]
        yield tuple(o_swapped), tuple(x_swapped)


def commutations(markers: Markers):
    """Every grid one commutation away."""
    yield from _row_commutations(markers)
    for swapped in _row_commutations(_transposed(markers)):
        yield _transposed(swapped)


def _destabilizable_within(markers: Markers, depth: int) -> Markers | None:
    """A grid at most ``depth`` commutations away that allows a destabilization, or None."""
    seen = {markers}
    layer = deque([markers])
    for _ in range(depth):
        following = deque()
        for current in layer:
            for neighbour in commutations(current):
                if neighbour in seen:
                    continue
                if destabilized(neighbour) is not None:
                    return neighbour
                seen.add(neighbour)
                following.append(neighbour)
        layer = following
    return None


def simplified(grid: GridDiagram, depth: int) -> GridDiagram:
    """The grid made as small as destabilizations reachable through at most ``depth``
    commutations at a time allow."""
    markers = (tuple(grid.O), tuple(grid.X))
    while True:
        smaller = destabilized(markers)
        if smaller is None:
            reachable = _destabilizable_within(markers, depth)
            if reachable is None:
                return GridDiagram(list(markers[0]), list(markers[1]))
            smaller = destabilized(reachable)
        markers = smaller
