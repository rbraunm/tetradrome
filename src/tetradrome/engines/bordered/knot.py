# SPDX-License-Identifier: Apache-2.0

"""Knot Floer invariants of a knot from its Morse presentation by the bordered algebras of
Ozsvath and Szabo: HFK-hat (arXiv:1603.06559, section 11.2), and the knot Floer complex over
F2[U, V]/(UV) with tau, nu and epsilon (arXiv:1707.00597, section 8.2, in the curved form of
its section 13 and arXiv:1912.01657).

The diagram is read from the top down. The global maximum is the maximum bimodule on the empty
slice; every slab below it, down to the global minimum, is a DA bimodule tensored onto the type D
structure above it: a crossing ``("cross", i, over)`` is P^{i+1} when the strand from the lower
left passes over and N^{i+1} otherwise, a maximum ``("max", i)`` is Omega^{i+1}, and a minimum
``("min", i)`` is mho^{i+1}. That is built as mho^1 after the two positive crossings of (9.4)
for each step to the left, or as the rotation of mho^1 (Lemma 10.1) after the two positive
crossings of the rotated (9.4) for each step to the right, whichever end is nearer. The global
minimum, the marked point, is the terminal module.

``hfk_hat`` runs this over the algebras B(m, k, S) of 1603.06559. ``knot_complex`` runs it over
the curved algebras of 1707.00597, which carry the matching of each slice in place of the C's,
and pairs the result with the terminal module over F2[U, V]/(UV).

Each slab's type D structure is, by default, cancelled to a smaller homotopy equivalent one, and
the structure carried down is checked (``TypeD.check``); ``cancel=False`` keeps every generator,
the faithful reference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ...algebra.reduce_reference import homology
from ...diagrams.model import PDCode
from ...diagrams.morse import MorseDiagram
from ...errors import TetradromeError
from .algebra import Algebra, CurvedAlgebra
from .cancellation import cancelled
from .crossing import CrossingBimodule
from .curved_minimum import CurvedMinimumBimodule, curved_minimum_weight_budget
from .knot_complex import KnotComplex
from .maximum import MaximumBimodule
from .minimum import MinimumBimodule, minimum_weight_budget
from .rotation import RotatedBimodule, rotated_algebra
from .tensor import box
from .terminal import hat_complexes
from .type_d import TypeD


@dataclass(frozen=True)
class MorseSource:
    """What a bordered result was computed from: the reduced crossing diagram and the Morse
    presentation cut from it."""

    diagram: PDCode
    morse: MorseDiagram

    @property
    def widest(self) -> int:
        return max(self.morse.widths())


def _step(bimodule, typed: TypeD, cancel: bool) -> TypeD:
    typed = box(bimodule, typed)
    if cancel:
        typed = cancelled(typed)
    typed.check()
    return typed


def _minimum_1603(algebra: Algebra, typed: TypeD):
    return MinimumBimodule(algebra, minimum_weight_budget(typed))


def _minimum_curved(algebra: CurvedAlgebra, typed: TypeD):
    return CurvedMinimumBimodule(algebra, curved_minimum_weight_budget(typed))


def _minimum(typed: TypeD, strand: int, cancel: bool, cup) -> TypeD:
    last = typed.algebra.strands - 2
    if strand <= last - strand:
        for crossing in range(strand - 1, -1, -1):
            for strands in (crossing, crossing + 1):
                typed = _step(CrossingBimodule(typed.algebra, strands, True), typed, cancel)
        return _step(cup(typed.algebra, typed), typed, cancel)
    for crossing in range(strand, last):
        for strands in (crossing + 1, crossing):
            typed = _step(CrossingBimodule(typed.algebra, strands, True), typed, cancel)
    inner = cup(rotated_algebra(typed.algebra), typed)
    return _step(RotatedBimodule(inner, typed.algebra), typed, cancel)


def _descend(morse: MorseDiagram, upward: list, cancel: bool, top: Algebra, cup) -> TypeD:
    pieces = morse.pieces
    widths = morse.widths()
    if pieces[0] != ("min", 0) or pieces[-1] != ("max", 0) or 0 in widths[1:-1]:
        raise TetradromeError("the presentation is not one knot between one global minimum and "
                              "one global maximum.")
    if [len(level) for level in upward] != widths:
        raise TetradromeError("the orientation does not match the presentation's levels.")
    typed = TypeD(top)
    typed.add_generator(0, 0, 0)
    for level in range(len(pieces) - 1, 0, -1):
        kind, at = pieces[level][0], pieces[level][1]
        if kind == "max":
            typed = _step(MaximumBimodule(typed.algebra, at, upward[level][at]), typed, cancel)
        elif kind == "min":
            typed = _minimum(typed, at, cancel, cup)
        else:
            typed = _step(CrossingBimodule(typed.algebra, at, positive=pieces[level][2]),
                          typed, cancel)
        expected = frozenset(s for s, up in enumerate(upward[level]) if up)
        if typed.algebra.upward != expected:
            raise TetradromeError(f"below piece {level} the strands pointing up are "
                                  f"{sorted(typed.algebra.upward)}, not {sorted(expected)}.")
    return typed


def type_d_above_minimum(morse: MorseDiagram, upward: list, *, cancel: bool = True) -> TypeD:
    """The type D structure over B(m, k, S) of everything above the global minimum, oriented by
    ``upward`` (for each level, as ``MorseDiagram.directions`` gives it, whether each strand
    points up)."""
    return _descend(morse, upward, cancel, Algebra(0, 0, frozenset()), _minimum_1603)


def curved_type_d_above_minimum(morse: MorseDiagram, upward: list, *,
                                cancel: bool = True) -> TypeD:
    """The curved type D structure of everything above the global minimum (arXiv:1707.00597,
    Proposition 8.2, in the curved form), oriented by ``upward`` as for
    ``type_d_above_minimum``."""
    return _descend(morse, upward, cancel, CurvedAlgebra(0, 0, frozenset(), ()), _minimum_curved)


def homology_of(typed: TypeD) -> dict:
    """HFK-hat ``{(maslov, alexander): rank}`` from the type D structure above the global
    minimum."""
    ranks: dict = defaultdict(int)
    for alexander, complex_ in hat_complexes(typed).items():
        for degree, rank in homology(complex_).items():
            ranks[(-degree, alexander)] += rank
    return dict(ranks)


def hfk_hat(morse: MorseDiagram, *, cancel: bool = True) -> dict:
    """HFK-hat of the knot ``morse`` draws, ``{(maslov, alexander): rank}``."""
    return homology_of(type_d_above_minimum(morse, morse.directions(), cancel=cancel))


def knot_complex(morse: MorseDiagram, *, cancel: bool = True) -> KnotComplex:
    """The knot Floer complex over F2[U, V]/(UV) of the knot ``morse`` draws."""
    return KnotComplex.from_type_d(
        curved_type_d_above_minimum(morse, morse.directions(), cancel=cancel))


def tau(morse: MorseDiagram) -> int:
    """Ozsvath and Szabo's tau of the knot ``morse`` draws."""
    return knot_complex(morse).tau()


def nu(morse: MorseDiagram) -> int:
    """Ozsvath and Szabo's nu of the knot ``morse`` draws."""
    return knot_complex(morse).nu()


def epsilon(morse: MorseDiagram) -> int:
    """Hom's epsilon of the knot ``morse`` draws."""
    return knot_complex(morse).epsilon()
