# SPDX-License-Identifier: Apache-2.0

"""Removing the crossings a knot diagram does not need: kinks and nugatory crossings.

A kink is a crossing one of whose arcs returns to it; a first Reidemeister move removes it. A
nugatory crossing is one whose removal disconnects the diagram: the two sides meet only there,
so turning one side over (a half turn about an axis in the plane through the crossing) undoes
it. Neither changes the knot. A diagram with neither is reduced: as a plane graph (crossings as
vertices, arcs as edges) it has no loops and no cut vertices.

PD convention as everywhere: crossing [a, b, c, d] lists its arcs counterclockwise from the
incoming understrand, so a, c are the understrand and b, d the overstrand. The reduced code is
relabelled 1..2n along the knot with the understrand running a -> c = a + 1 (mod 2n), KnotInfo's
convention.
"""
from __future__ import annotations

from .model import PDCode


def _slots(crossings) -> dict:
    """Each arc label -> the two (crossing, position) slots it joins."""
    ends: dict = {}
    for index, crossing in enumerate(crossings):
        for position, label in enumerate(crossing):
            ends.setdefault(label, []).append((index, position))
    return ends


def _other_end(ends, crossings, index, position):
    first, second = ends[crossings[index][position]]
    return second if first == (index, position) else first


def _rename(crossings, old, new) -> None:
    for crossing in crossings:
        for position in range(4):
            if crossing[position] == old:
                crossing[position] = new


def _components(crossings) -> int:
    """How many closed curves the diagram is: strands continue straight through a crossing."""
    ends = _slots(crossings)
    unseen = {(index, position) for index in range(len(crossings)) for position in range(4)}
    count = 0
    while unseen:
        count += 1
        start = min(unseen)
        index, position = start
        while True:
            unseen.discard((index, position))
            leaving = (index, (position + 2) % 4)
            unseen.discard(leaving)
            index, position = _other_end(ends, crossings, *leaving)
            if (index, position) == start:
                break
    return count


def _remove_kink(crossings) -> bool:
    for index, crossing in enumerate(crossings):
        for position in range(4):
            if crossing[position] != crossing[(position + 1) % 4]:
                continue
            kept = crossing[(position + 2) % 4], crossing[(position + 3) % 4]
            del crossings[index]
            if kept[0] != kept[1]:          # otherwise the crossing was the whole diagram
                _rename(crossings, kept[1], kept[0])
            return True
    return False


def _remove_nugatory(crossings) -> bool:
    ends = _slots(crossings)
    for index, crossing in enumerate(crossings):
        component = {}
        for position in range(4):
            start = _other_end(ends, crossings, index, position)[0]
            if start in component:
                continue
            label = len(set(component.values()))
            stack = [start]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component[current] = label
                for neighbour_position in range(4):
                    neighbour = _other_end(ends, crossings, current, neighbour_position)[0]
                    if neighbour != index and neighbour not in component:
                        stack.append(neighbour)
        side = [component[_other_end(ends, crossings, index, position)[0]]
                for position in range(4)]
        if len(set(side)) == 1:
            continue
        # In a plane diagram the two sides each hold two cyclically adjacent positions.
        adjacent = [i for i in range(4) if side[i] == side[(i + 1) % 4] != side[(i + 2) % 4]]
        if not adjacent:
            raise ValueError(f"the PD code is not a plane diagram: the sides of crossing "
                             f"{tuple(crossing)} interleave around it.")
        first = adjacent[0]
        joined_to = {crossing[(first + 2) % 4]: crossing[first],
                     crossing[(first + 3) % 4]: crossing[(first + 1) % 4]}
        far_side = side[(first + 2) % 4]
        for other, other_crossing in enumerate(crossings):
            # Turning the far side over mirrors its plane picture and swaps every crossing's
            # over and under strands: reversing the counterclockwise list does both.
            if other != index and component[other] == far_side:
                other_crossing.reverse()
        del crossings[index]
        for old, new in joined_to.items():
            _rename(crossings, old, new)
        return True
    return False


def _relabelled(crossings, start) -> PDCode:
    """Labels 1..2n in the order the knot passes them, from the arc entering the slot
    ``start = (crossing, position)``; crossings numbered in the order the walk first meets them,
    each listed from its incoming understrand."""
    ends = _slots(crossings)
    label_of = {}
    understrand_entry = {}
    met = []
    index, position = start
    label = 0
    while True:
        if index not in met:
            met.append(index)
        if position % 2 == 0:
            understrand_entry[index] = position
        leaving = (index, (position + 2) % 4)
        label += 1
        arriving = _other_end(ends, crossings, *leaving)
        label_of[leaving] = label_of[arriving] = label
        index, position = arriving
        if (index, position) == start:
            break
    relabelled = []
    for index in met:
        entry = understrand_entry[index]
        relabelled.append(tuple(label_of[(index, (entry + step) % 4)] for step in range(4)))
    return tuple(relabelled)


def relabellings(pd: PDCode) -> list:
    """The diagram relabelled from every arc in both directions: each a code of the same
    diagram, first in canonical order (the least)."""
    crossings = [list(crossing) for crossing in pd]
    codes = {_relabelled(crossings, (index, position))
             for index in range(len(crossings)) for position in range(4)}
    return sorted(codes)


def reduced(pd: PDCode) -> PDCode:
    """The knot's diagram without kinks or nugatory crossings, in canonical labelling: relabelled
    along the knot, the least such code. Two codes of one diagram that differ only in labelling
    reduce to the same code. An unknot diagram may reduce to no crossings at all, the empty code.
    A code of several closed curves is a link, not a knot, and raises."""
    crossings = [list(crossing) for crossing in pd]
    curves = _components(crossings)
    if curves != 1:
        raise ValueError(f"the PD code describes a {curves}-component link, not a knot.")
    crossings = [list(crossing) for crossing in relabellings(pd)[0]]
    while _remove_kink(crossings) or _remove_nugatory(crossings):
        pass
    return relabellings(crossings)[0] if crossings else ()
