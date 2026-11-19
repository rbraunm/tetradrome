# SPDX-License-Identifier: Apache-2.0

"""The global minimum: the terminal module paired with the type D structure above it
(arXiv:1603.06559, sections 9.4 and 11.2).

Above the global minimum the algebra is B(2, 1, S). The associated graded terminal module has
generators X over the idempotent {1} and Y over the other, and only idempotents act on it, so
its box tensor product with a type D structure is the complex on X (x) y with differential the
arrows whose coefficient is the identity. That complex is C-hat of the diagram; its generators
carry the (Maslov, Alexander) bigrading of the type D structure. Every type D structure here
avoids positions 0 and 2, so only X appears.
"""
from __future__ import annotations

from collections import defaultdict

from ...algebra.complex import GradedComplex
from ...errors import TetradromeError
from .type_d import TypeD

_ONE = 1 << 1


def hat_complexes(typed: TypeD) -> dict:
    """C-hat split by Alexander grading: ``{alexander: GradedComplex}``, the complex's degree
    being minus the Maslov grading so that its differential raises it."""
    algebra = typed.algebra
    if algebra.strands != 2 or algebra.occupied != 1:
        raise TetradromeError(f"the terminal module pairs with B(2, 1, S), not {algebra}.")
    identity = algebra.identity(_ONE)
    members: dict = defaultdict(lambda: defaultdict(list))
    place = {}
    for generator in range(len(typed)):
        if typed.idempotents[generator] != _ONE:
            raise TetradromeError("a generator above the global minimum is off the idempotent {1}.")
        if typed.alexander[generator] % 2:
            raise TetradromeError("a generator above the global minimum has a half-integral "
                                  "Alexander grading.")
        alexander, degree = typed.alexander[generator] // 2, -typed.maslov[generator]
        place[generator] = (alexander, degree, len(members[alexander][degree]))
        members[alexander][degree].append(generator)
    complexes = {}
    for alexander, by_degree in members.items():
        maps = {}
        for degree, sources in by_degree.items():
            columns = []
            for source in sources:
                column = []
                for target, coefficient in typed.arrows[source].items():
                    if identity in coefficient:
                        target_alexander, target_degree, row = place[target]
                        if (target_alexander, target_degree) != (alexander, degree + 1):
                            raise TetradromeError("an identity arrow does not lower the Maslov "
                                                  "grading by one at a fixed Alexander grading.")
                        column.append(row)
                columns.append(column)
            if any(columns):
                maps[degree] = columns
        dims = {degree: len(sources) for degree, sources in by_degree.items()}
        complexes[alexander] = GradedComplex(dims, maps)
    return complexes
