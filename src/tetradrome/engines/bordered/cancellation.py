# SPDX-License-Identifier: Apache-2.0

"""Cancellation of a type D structure to a smaller homotopy equivalent one.

An arrow x -> y whose coefficient is exactly the idempotent of x is cancelled: x and y are
removed and every z with an arrow a to y gains a.b (x) w for each term b (x) w of delta^1(x)
with w neither x nor y. This is the homological perturbation lemma for the retraction that
contracts x -> y (homotopy y -> x); since the coefficient is the idempotent itself, its
algebra differential vanishes and the series stops after one step. The result is homotopy
equivalent to the input as a graded type D structure, so every box tensor product and the
final homology are unchanged. Cancelling repeats until no such arrow is left.
"""
from __future__ import annotations

from .type_d import TypeD


def cancelled(typed: TypeD) -> TypeD:
    """``typed`` with every identity arrow cancelled."""
    algebra = typed.algebra
    arrows = [{target: set(coefficient) for target, coefficient in out.items()}
              for out in typed.arrows]
    incoming = [set() for _ in range(len(typed))]
    for source, out in enumerate(arrows):
        for target in out:
            incoming[target].add(source)
    alive = [True] * len(typed)
    identities = {state: algebra.identity(state) for state in set(typed.idempotents)}

    def is_identity(source, target):
        coefficient = arrows[source].get(target)
        return (coefficient is not None and len(coefficient) == 1
                and identities[typed.idempotents[source]] in coefficient)

    pending = [(source, target) for source, out in enumerate(arrows) for target in out
               if is_identity(source, target)]
    while pending:
        source, target = pending.pop()
        if not (alive[source] and alive[target] and is_identity(source, target)):
            continue
        onward = [(element, end) for end, coefficient in arrows[source].items()
                  if end != source and end != target for element in coefficient]
        for before in incoming[target]:
            if before == source or before == target:
                continue
            out = arrows[before]
            for first in out[target]:
                for second, end in onward:
                    product = algebra.multiply(first, second)
                    if product is None:
                        continue
                    coefficient = out.setdefault(end, set())
                    coefficient ^= {product}
                    if not coefficient:
                        del out[end]
                        incoming[end].discard(before)
                    else:
                        incoming[end].add(before)
                        if len(coefficient) == 1 and is_identity(before, end):
                            pending.append((before, end))
        for gone in (source, target):
            alive[gone] = False
            for end in arrows[gone]:
                incoming[end].discard(gone)
            for before in incoming[gone]:
                arrows[before].pop(gone, None)
            arrows[gone] = {}
            incoming[gone] = set()
    reduced = TypeD(algebra)
    renumbered = {}
    for generator in range(len(typed)):
        if alive[generator]:
            renumbered[generator] = reduced.add_generator(
                typed.idempotents[generator], typed.maslov[generator], typed.alexander[generator])
    for generator, out in enumerate(arrows):
        if alive[generator]:
            reduced.arrows[renumbered[generator]] = {
                renumbered[end]: coefficient for end, coefficient in out.items()}
    return reduced
