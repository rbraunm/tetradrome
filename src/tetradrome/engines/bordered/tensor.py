# SPDX-License-Identifier: Apache-2.0

"""The box tensor product of a DA bimodule with a type D structure (arXiv:1603.06559, sections
2.5 and 2.7).

A bimodule gives its generators over each right idempotent, their gradings, and its operations
as a transducer: ``start(generator)`` is a state with no inputs, ``advance(state, element)``
feeds one more input (None when no operation takes such a sequence), ``accepts_more(state)``
says whether any further input can be fed, and ``emit(state)`` is delta^1 on the inputs fed so
far, a set of ``(element, generator)``. The product follows every
path of the type D structure from each generator, feeding the elements on the path to the
bimodule. States reached by two paths that end at the same generator are merged over F2, since
they have the same future. ``input_limit`` bounds the number of inputs; a longer live path
raises.
"""
from __future__ import annotations

from typing import NamedTuple

from ...errors import TetradromeError
from .type_d import TypeD


class Generator(NamedTuple):
    kind: str
    left: int
    right: int


def box(bimodule, typed: TypeD) -> TypeD:
    """``bimodule`` box-tensored with ``typed``: a type D structure over the bimodule's output."""
    if bimodule.upper != typed.algebra:
        raise TetradromeError(f"the bimodule takes {bimodule.upper}, the type D structure is over "
                              f"{typed.algebra}.")
    product = TypeD(bimodule.lower)
    index: dict = {}
    for source in range(len(typed)):
        for generator in bimodule.generators(typed.idempotents[source]):
            if generator.right != typed.idempotents[source]:
                raise TetradromeError(f"generator {generator} does not sit over idempotent "
                                      f"{typed.idempotents[source]}.")
            index[(generator, source)] = product.add_generator(
                generator.left, bimodule.maslov(generator) + typed.maslov[source],
                bimodule.alexander(generator) + typed.alexander[source])
    limit = bimodule.input_limit(typed)
    outgoing = [list(typed.terms(source)) for source in range(len(typed))]
    for (generator, source), start in index.items():
        frontier = {(bimodule.start(generator), source)}
        for _ in range(limit + 1):
            following: set = set()
            for state, current in frontier:
                for element, target in bimodule.emit(state):
                    end = index.get((target, current))
                    if end is None:
                        raise TetradromeError(f"an operation ends at {target}, which is not "
                                              f"over the idempotent of generator {current}.")
                    product.toggle(start, element, end)
                if not bimodule.accepts_more(state):
                    continue
                for element, onward in outgoing[current]:
                    advanced = bimodule.advance(state, element)
                    if advanced is not None:
                        following ^= {(advanced, onward)}
            frontier = following
            if not frontier:
                break
        else:
            raise TetradromeError(f"paths of more than {limit} inputs stay live in the tensor "
                                  f"product: the diagram is not a knot or a bound is wrong.")
    return product
