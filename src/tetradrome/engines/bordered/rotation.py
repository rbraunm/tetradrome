# SPDX-License-Identifier: Apache-2.0

"""The rotation R through a vertical axis (arXiv:1603.06559, sections 3.6 and 10).

R is the algebra isomorphism B(m, k, S) -> B(m, k, rho'(S)) that reflects positions p to m - p
and strands s to m - 1 - s, so L moves become R moves. Conjugating a bimodule by R gives the
bimodule of the rotated slab (Lemma 10.1); ``RotatedBimodule`` does that, which turns mho^1 at
the left end into the minimum at the right end.
"""
from __future__ import annotations

from ...errors import TetradromeError
from .algebra import Algebra, Element, pack, unpack
from .tensor import Generator


def rotated_state(state: int, strands: int) -> int:
    rotated = 0
    for position in range(strands + 1):
        if state >> position & 1:
            rotated |= 1 << strands - position
    return rotated


def rotated_strands(mask: int, strands: int) -> int:
    rotated = 0
    for strand in range(strands):
        if mask >> strand & 1:
            rotated |= 1 << strands - 1 - strand
    return rotated


def rotated_algebra(algebra: Algebra) -> Algebra:
    return Algebra(algebra.strands, algebra.occupied,
                   frozenset(algebra.strands - 1 - s for s in algebra.upward))


def rotated_element(element: Element, strands: int) -> Element:
    return Element(rotated_state(element.left, strands), rotated_state(element.right, strands),
                   pack(unpack(element.weights, strands)[::-1]),
                   rotated_strands(element.c_strands, strands))


class RotatedBimodule:
    """R^-1 (x) inner (x) R, for ``inner`` built over the rotation of ``upper``."""

    def __init__(self, inner, upper: Algebra):
        if inner.upper != rotated_algebra(upper):
            raise TetradromeError(f"{inner.upper} is not the rotation of {upper}.")
        self.inner = inner
        self.upper = upper
        self.lower = rotated_algebra(inner.lower)

    def _turned(self, generator: Generator) -> Generator:
        """A generator seen from the other side of the rotation (R is an involution)."""
        return Generator(generator.kind, rotated_state(generator.left, self.lower.strands),
                         rotated_state(generator.right, self.upper.strands))

    def generators(self, right: int) -> list:
        return [self._turned(generator) for generator in
                self.inner.generators(rotated_state(right, self.upper.strands))]

    def maslov(self, generator: Generator) -> int:
        return self.inner.maslov(self._turned(generator))

    def alexander(self, generator: Generator) -> int:
        return self.inner.alexander(self._turned(generator))

    def input_limit(self, typed) -> int:
        return self.inner.input_limit(typed)

    def start(self, generator: Generator):
        return self.inner.start(self._turned(generator))

    def accepts_more(self, state) -> bool:
        return self.inner.accepts_more(state)

    def advance(self, state, element: Element):
        return self.inner.advance(state, rotated_element(element, self.upper.strands))

    def emit(self, state) -> set:
        return {(rotated_element(element, self.lower.strands), self._turned(generator))
                for element, generator in self.inner.emit(state)}
