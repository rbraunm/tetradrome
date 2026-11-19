# SPDX-License-Identifier: Apache-2.0

"""The algebra B(m, k, S) of Ozsvath and Szabo (arXiv:1603.06559, section 3).

Strands are numbered 0 to m - 1 from the left (the paper's strands 1 to m) and the gaps between
them, the positions, 0 to m as in the paper: strand s lies between positions s and s + 1. An
idempotent state is a set of k positions, held as a bit mask. ``upward`` is the paper's S, the
strands oriented upwards, which carry the elements C_s.

A pure element is stored as ``Element(left, right, weights, c_strands)``: its two idempotent states,
its weight vector doubled so that it is integral, and the strands s whose C_s divide it. By the
paper's Proposition 3.9 these determine it; the U-monomial is the weight less the minimal
relative weight of the two states and the weights of the C's. ``element`` returns None for a
weight that no element has or an element that the relations kill.

The doubled weights are packed into one integer, ``FIELD`` bits per strand, strand s in bits
FIELD * s upwards (``pack``, ``unpack``), so that adding weights is adding integers
and the relations are tested with masks. A weight must stay below ``LIMIT``, which keeps every
field and every sum of fields inside its bits; a larger one raises.
"""
from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

from ...errors import TetradromeError

FIELD = 16
FIELD_MASK = (1 << FIELD) - 1
LIMIT = 1 << 11
_GUARD = 1 << FIELD - 1


class Element(NamedTuple):
    left: int
    right: int
    weights: int
    c_strands: int


def pack(values) -> int:
    """Doubled weights, strand by strand, as one integer."""
    packed = 0
    for strand, value in enumerate(values):
        if not 0 <= value < LIMIT:
            raise TetradromeError(f"weight {value} is outside the packed range.")
        packed |= value << FIELD * strand
    return packed


def unpack(weights: int, strands: int) -> tuple:
    return tuple(weights >> FIELD * strand & FIELD_MASK for strand in range(strands))


def field_sum(weights: int) -> int:
    """The sum of the fields: 2^FIELD is 1 modulo 2^FIELD - 1, and the sum is smaller."""
    return weights % FIELD_MASK


def positions(state: int) -> list:
    """The positions of an idempotent state, increasing."""
    found = []
    position = 0
    while state:
        if state & 1:
            found.append(position)
        state >>= 1
        position += 1
    return found


def state_of(chosen) -> int:
    """The idempotent state holding the given positions."""
    state = 0
    for position in chosen:
        state |= 1 << position
    return state


def _every(strands: int, value: int) -> int:
    return sum(value << FIELD * strand for strand in range(strands))


class Algebra:
    """B(strands, occupied, upward) with its relations, products, differential and gradings."""

    def __init__(self, strands: int, occupied: int, upward: frozenset):
        if not 0 <= occupied <= strands + 1:
            raise TetradromeError(f"B({strands}, {occupied}) has no idempotent states.")
        if any(not 0 <= strand < strands for strand in upward):
            raise TetradromeError(f"upward strands {sorted(upward)} are not among {strands}.")
        if strands * LIMIT >= FIELD_MASK:
            raise TetradromeError(f"{strands} strands do not fit the packed weights.")
        self.strands = strands
        self.occupied = occupied
        self.upward = frozenset(upward)
        self.upward_mask = state_of(self.upward)
        self.upward_fields = sum(FIELD_MASK << FIELD * s for s in self.upward)
        self._guards = _every(strands, _GUARD)
        self._overflow = _every(strands, FIELD_MASK ^ (LIMIT - 1)) | -1 << FIELD * strands
        self._units = _every(strands, 1)
        self._below_two = _every(strands, _GUARD - 2)
        self._levels: dict = {}
        self._pairs: dict = {}
        self._spread: dict = {}

    def __eq__(self, other) -> bool:
        return (isinstance(other, Algebra) and self.strands == other.strands
                and self.occupied == other.occupied and self.upward == other.upward)

    def __hash__(self) -> int:
        return hash((self.strands, self.occupied, self.upward))

    def __repr__(self) -> str:
        return f"B({self.strands}, {self.occupied}, {sorted(self.upward)})"

    def states(self) -> list:
        """Every idempotent state."""
        return [state_of(chosen) for chosen in combinations(range(self.strands + 1), self.occupied)]

    def levels(self, state: int) -> tuple:
        """The weight v of a state (the paper's (3.3)): for each strand, the occupied positions
        to its right."""
        found = self._levels.get(state)
        if found is None:
            occupied = positions(state)
            if len(occupied) != self.occupied or (occupied and occupied[-1] > self.strands):
                raise TetradromeError(f"{occupied} is not an idempotent state of {self}.")
            found = tuple(sum(1 for position in occupied if position > strand)
                          for strand in range(self.strands))
            self._levels[state] = found
        return found

    def pair(self, left: int, right: int) -> tuple:
        """``(far, relative, intervals)`` for two states: whether they are far (definition 3.5),
        their doubled minimal relative weight, packed, and their generating intervals
        (definition 3.6), each as the mask of the guard bits of its strands."""
        found = self._pairs.get((left, right))
        if found is None:
            far = any(abs(first - second) > 1
                      for first, second in zip(positions(left), positions(right)))
            relative = tuple(abs(first - second)
                             for first, second in zip(self.levels(left), self.levels(right)))
            common = left & right
            free = [position for position in range(self.strands + 1) if not common >> position & 1]
            intervals = tuple(sum(_GUARD << FIELD * strand for strand in range(low, high))
                              for low, high in zip(free, free[1:]) if not any(relative[low:high]))
            found = (far, pack(relative), intervals)
            self._pairs[(left, right)] = found
        return found

    def _doubled_c_weights(self, c_strands: int) -> int:
        """Twice the weight of the C's in ``c_strands``, packed."""
        found = self._spread.get(c_strands)
        if found is None:
            if c_strands & ~self.upward_mask:
                strays = positions(c_strands & ~self.upward_mask)
                raise TetradromeError(f"C's on strands {strays} of {self}, which are not upward.")
            found = sum(2 << FIELD * strand for strand in positions(c_strands))
            self._spread[c_strands] = found
        return found

    def _killed(self, weights: int, c_strands: int, intervals: tuple) -> bool:
        """Whether the U-monomial is divisible by the product over a generating interval, on
        whose strands the relative weight vanishes."""
        if c_strands:
            weights -= self._doubled_c_weights(c_strands)
        flags = (weights + self._below_two) & self._guards
        for interval in intervals:
            if flags & interval == interval:
                return True
        return False

    def element(self, left: int, right: int, weights: int, c_strands: int) -> Element | None:
        """The pure element with these states, doubled weights and C's, or None if there is no
        nonzero one."""
        if weights & self._overflow:
            raise TetradromeError(f"weights {unpack(weights, self.strands)} are outside the "
                                  f"packed range.")
        found = self._pairs.get((left, right))
        far, relative, intervals = found if found is not None else self.pair(left, right)
        if far:
            return None
        exponents = weights + self._guards - relative
        if c_strands:
            exponents -= self._doubled_c_weights(c_strands)
        if exponents & self._guards != self._guards or exponents & self._units:
            return None
        if intervals and self._killed(weights, c_strands, intervals):
            return None
        return Element(left, right, weights, c_strands)

    def exponents(self, element: Element) -> tuple:
        """The exponents of the U-monomial of a pure element."""
        relative = self.pair(element.left, element.right)[1]
        doubled = element.weights - relative - self._doubled_c_weights(element.c_strands)
        return tuple(value // 2 for value in unpack(doubled, self.strands))

    def multiply(self, first: Element, second: Element) -> Element | None:
        """The product, first then second, or None when it vanishes."""
        left, middle, first_weights, first_c_strands = first
        start, right, second_weights, second_c_strands = second
        if middle != start or first_c_strands & second_c_strands:
            return None
        found = self._pairs.get((left, right))
        far, _, intervals = found if found is not None else self.pair(left, right)
        if far:
            return None
        # The exponents of a product are those of the factors plus the nonnegative even
        # relative-weight defect of the three states, so only the relations can kill it.
        weights = first_weights + second_weights
        if weights & self._overflow:
            raise TetradromeError(f"weights {unpack(weights, self.strands)} are outside the "
                                  f"packed range.")
        c_strands = first_c_strands | second_c_strands
        if intervals and self._killed(weights, c_strands, intervals):
            return None
        return Element(left, right, weights, c_strands)

    def differential(self, element: Element) -> set:
        """The differential, d C_s = U_s, as a set of pure elements (a sum over F2)."""
        terms: set = set()
        c_strands = element.c_strands
        while c_strands:
            bit = c_strands & -c_strands
            c_strands ^= bit
            term = self.element(element.left, element.right, element.weights,
                                element.c_strands ^ bit)
            if term is not None:
                terms ^= {term}
        return terms

    def maslov(self, element: Element) -> int:
        """The Maslov grading (the paper's (3.9))."""
        return bin(element.c_strands).count("1") - field_sum(element.weights & self.upward_fields)

    def alexander(self, element: Element) -> int:
        """Twice the Alexander grading (the paper's (3.8))."""
        return field_sum(element.weights) - 2 * field_sum(element.weights & self.upward_fields)

    def identity(self, state: int) -> Element:
        return Element(state, state, 0, 0)

    def right_move(self, state: int, strand: int) -> Element | None:
        """R across ``strand``: the coordinate at its left moves to its right."""
        if not state >> strand & 1 or state >> strand + 1 & 1:
            return None
        return self.element(state, state ^ (3 << strand), 1 << FIELD * strand, 0)

    def left_move(self, state: int, strand: int) -> Element | None:
        """L across ``strand``: the coordinate at its right moves to its left."""
        if state >> strand & 1 or not state >> strand + 1 & 1:
            return None
        return self.element(state, state ^ (3 << strand), 1 << FIELD * strand, 0)

    def u(self, state: int, strand: int) -> Element | None:
        return self.element(state, state, 2 << FIELD * strand, 0)

    def c(self, state: int, strand: int) -> Element | None:
        return self.element(state, state, 2 << FIELD * strand, 1 << strand)
