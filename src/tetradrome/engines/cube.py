"""The resolution-cube skeleton: a PD diagram, a 0/1 choice at each crossing, and
the circles that result.

This is shared scaffolding. The Kauffman bracket / Jones (invariants/jones.py) needs
only the *number* of circles per state; the native Khovanov engine (later) needs the
circles themselves and how they merge/split between adjacent states. So `resolve`
returns the circles as sets of arc labels, not just a count.

Smoothing convention (validated against KnotInfo via the Jones polynomial across every
knot through 11 crossings, chirality included):

    crossing PD [a, b, c, d], arc labels counterclockwise
      bit 0  ->  A-smoothing: joins (b,c) and (d,a)
      bit 1  ->  B-smoothing: joins (a,b) and (c,d)

A state is a sequence of bits, one per crossing, in PD order.
"""
from __future__ import annotations

from ..diagrams.model import PDCode

State = tuple[int, ...]


def _smoothing_pairs(crossing: tuple[int, int, int, int], bit: int):
    a, b, c, d = crossing
    if bit == 0:
        return (b, c), (d, a)  # A-smoothing
    return (a, b), (c, d)      # B-smoothing


def resolve(pd: PDCode, state) -> tuple[frozenset[int], ...]:
    """The circles of one fully-smoothed state, each as a frozenset of arc labels.

    Each arc label sits on exactly one circle, so the circles partition the labels;
    the number of circles is `len(resolve(pd, state))`.
    """
    if len(state) != len(pd):
        raise ValueError(f"state has {len(state)} bits but the diagram has {len(pd)} crossings.")

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    labels: set[int] = set()
    for crossing in pd:
        labels.update(crossing)
    for label in labels:
        parent.setdefault(label, label)

    for crossing, bit in zip(pd, state):
        (p, q), (r, s) = _smoothing_pairs(crossing, bit)
        union(p, q)
        union(r, s)

    groups: dict[int, set[int]] = {}
    for label in labels:
        groups.setdefault(find(label), set()).add(label)
    return tuple(frozenset(g) for g in groups.values())


def circle_count(pd: PDCode, state) -> int:
    """Number of circles in a state (the only thing the Kauffman bracket needs)."""
    return len(resolve(pd, state))


def states(n: int):
    """Iterate every 0/1 state of an n-crossing diagram, low crossing index first.

    There are 2**n of them; this is the cube-of-resolutions explosion, expected and
    fine at the small crossing numbers Phase 0 validates against.
    """
    for s in range(1 << n):
        yield tuple((s >> i) & 1 for i in range(n))
