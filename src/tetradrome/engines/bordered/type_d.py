# SPDX-License-Identifier: Apache-2.0

"""Type D structures over B(m, k, S) (arXiv:1603.06559, section 2.4).

A generator has an idempotent state, a Maslov grading and a doubled Alexander grading. The map
delta^1 is held as arrows: for each source, its targets with the set of pure elements on each
arrow (a sum over F2). ``check`` verifies the structure relation, that every arrow lowers Maslov
by one and keeps the Alexander grading, and that every element runs between the idempotents of
its ends.
"""
from __future__ import annotations

from ...errors import TetradromeError
from .algebra import Algebra, Element


class TypeD:
    def __init__(self, algebra: Algebra):
        self.algebra = algebra
        self.idempotents: list = []
        self.maslov: list = []
        self.alexander: list = []
        self.arrows: list = []

    def __len__(self) -> int:
        return len(self.idempotents)

    def add_generator(self, idempotent: int, maslov: int, alexander: int) -> int:
        self.idempotents.append(idempotent)
        self.maslov.append(maslov)
        self.alexander.append(alexander)
        self.arrows.append({})
        return len(self.idempotents) - 1

    def toggle(self, source: int, element: Element, target: int) -> None:
        """Add ``element (x) target`` to delta^1 of ``source``, over F2."""
        coefficient = self.arrows[source].setdefault(target, set())
        coefficient ^= {element}
        if not coefficient:
            del self.arrows[source][target]

    def terms(self, source: int):
        """The terms ``(element, target)`` of delta^1 of ``source``."""
        for target, coefficient in self.arrows[source].items():
            for element in coefficient:
                yield element, target

    def check(self) -> None:
        """Raise unless this is a graded type D structure."""
        algebra = self.algebra
        idempotents, maslov, alexander = self.idempotents, self.maslov, self.alexander
        for source in range(len(self)):
            for element, target in self.terms(source):
                if element.left != idempotents[source] or element.right != idempotents[target]:
                    raise TetradromeError(f"an arrow {source} -> {target} carries an element "
                                          f"between the wrong idempotents.")
                if (maslov[source] - 1 != algebra.maslov(element) + maslov[target]
                        or alexander[source] != algebra.alexander(element) + alexander[target]):
                    raise TetradromeError(f"the arrow {source} -> {target} is not homogeneous.")
        outgoing = [list(self.terms(source)) for source in range(len(self))]
        multiply, differential = algebra.multiply, algebra.differential
        for source in range(len(self)):
            total: set = set()
            for element, middle in outgoing[source]:
                if element.c_strands:
                    for term in differential(element):
                        total ^= {(term, middle)}
                for following, target in outgoing[middle]:
                    product = multiply(element, following)
                    if product is not None:
                        total ^= {(product, target)}
            if total:
                raise TetradromeError(f"the type D structure relation fails at generator {source}.")
