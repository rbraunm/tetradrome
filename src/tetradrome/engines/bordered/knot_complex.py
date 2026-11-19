# SPDX-License-Identifier: Apache-2.0

"""The knot Floer complex over R = F2[U, V]/(UV) and the invariants read from it
(arXiv:1707.00597, sections 1, 8.2 and 11).

The complex is free over R on generators graded by (Maslov, Alexander). A term of the
differential from g to h has coefficient 1, U^u or V^v. A U^u arrow is horizontal: A(h) = A(g)
+ u and M(h) = M(g) - 1 + 2u. A V^v arrow is vertical: A(h) = A(g) - v and M(h) = M(g) - 1. A
unit arrow keeps A and lowers M by one; it is both. ``from_type_d`` pairs the curved type D
structure above the global minimum with the terminal module of section 8.2: U_s becomes U on
the strand s that points up just above the global minimum and V on the other.

tau and nu are the least s at which H_0 of the vertical subcomplex F_s = {A <= s},
respectively of A-hat_s = C{max(i, j - s) = 0}, maps onto H_0 of the vertical complex, which is
HF-hat of the three-sphere (Ozsvath and Szabo, "Knot Floer homology and the four-ball genus" and
"Knot Floer homology and rational surgeries"). Only horizontal and vertical arrows reach A-hat_s,
so it is the complex on all the generators with the unit arrows, the U arrows from A >= s and
the V arrows from A <= s, in which g stands for U^{A(g) - s} g or V^{s - A(g)} g. The paper's own
definitions (Definitions 1.5, 1.6 and 11.5) read the same groups from the horizontal complex
and the largest s; for a knot the symmetry exchanging U and V and negating A makes the two
equal, and both are computed and must agree. epsilon is (11.2), (tau - nu) - (tau' - nu'),
primed on the dual complex Hom(C, R), which is the mirror's (Proposition 10.3).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

from ...algebra.complex import GradedComplex
from ...algebra.reduce_f2_packed import pack_columns
from ...algebra.reduce_reference import f2_kernel, f2_rank, homology
from ...errors import TetradromeError
from .algebra import unpack
from .type_d import TypeD

_ONE = 1 << 1


def _reduced(columns) -> dict:
    """Pivots of the span of ``columns``, each keyed by its leading index."""
    pivots: dict = {}
    for column in columns:
        vector = set(column)
        while vector:
            lead = max(vector)
            if lead not in pivots:
                pivots[lead] = vector
                break
            vector ^= pivots[lead]
    return pivots


class _Localized:
    """C with one of U, V set to 0 and the other to 1: the F2 complex on the generators with the
    arrows whose coefficient ``keep`` admits, graded by ``degree``. Its homology must be F2 in
    degree 0, HF-hat of the three-sphere (Proposition 1.4)."""

    def __init__(self, complex_: KnotComplex, keep, degree):
        self.degree = [degree(maslov, alexander) for maslov, alexander in complex_.gradings]
        self.out = [[] for _ in complex_.gradings]
        for source, target, u, v in complex_.arrows:
            if keep(u, v):
                self.out[source].append(target)
        by_degree: dict = defaultdict(list)
        for generator, value in enumerate(self.degree):
            by_degree[value].append(generator)
        index = {generator: position for members in by_degree.values()
                 for position, generator in enumerate(members)}
        ranks = {value: f2_rank(pack_columns([index[t] for t in self.out[g]] for g in members))
                 for value, members in by_degree.items()}
        dimensions = {value: len(members) - ranks[value] - ranks.get(value + 1, 0)
                      for value, members in by_degree.items()}
        if {value: dimension for value, dimension in dimensions.items() if dimension} != {0: 1}:
            raise TetradromeError(f"the knot complex's reduction to HF-hat of the three-sphere "
                                  f"has homology {dimensions}, not F2 in degree 0.")
        self.index = {generator: index[generator] for generator in by_degree[0]}
        self.rows = {generator: index[generator] for generator in by_degree[-1]}
        self.boundaries = _reduced([[index[t] for t in self.out[g]] for g in by_degree[1]])


def _itself(generator: int) -> int:
    return generator


def _first(levels, good) -> int:
    """The first level that is ``good``."""
    for level in levels:
        if good(level):
            return level
    raise TetradromeError("no Alexander level carries the class of HF-hat.")


def _reaches(sources: list, out, rows: dict, image_of, target: _Localized) -> bool:
    """Whether some cycle on ``sources``, whose differentials ``out`` land in ``rows``, maps by
    ``image_of`` to a cycle of ``target`` in degree 0 that is not a boundary."""
    columns = []
    for source in sources:
        if any(t not in rows for t in out(source)):
            raise TetradromeError("an arrow of the knot complex leaves its degree.")
        columns.append(frozenset(rows[t] for t in out(source)))
    for relation in f2_kernel(columns):
        vector: set = set()
        for position in relation:
            image = image_of(sources[position])
            if image is not None:
                vector ^= {target.index[image]}
        while vector:
            pivot = target.boundaries.get(max(vector))
            if pivot is None:
                return True
            vector ^= pivot
    return False


@dataclass(frozen=True)
class KnotComplex:
    """``gradings[g]`` is ``(maslov, alexander)``; ``arrows`` holds the terms of the
    differential as ``(source, target, u, v)``, the coefficient U^u V^v with u v = 0."""

    gradings: tuple
    arrows: tuple

    def __post_init__(self):
        size = len(self.gradings)
        seen = set()
        for source, target, u, v in self.arrows:
            if not (0 <= source < size and 0 <= target < size) or u < 0 or v < 0 or u * v:
                raise TetradromeError(f"({source}, {target}, U^{u} V^{v}) is not an arrow of a "
                                      f"complex over F2[U, V]/(UV) on {size} generators.")
            if (source, target) in seen:
                raise TetradromeError(f"two arrows run from {source} to {target}.")
            seen.add((source, target))
            (maslov, alexander), (onward, reached) = self.gradings[source], self.gradings[target]
            if reached != alexander + u - v or onward != maslov - 1 + 2 * u:
                raise TetradromeError(f"the arrow {source} -> {target} is not homogeneous.")
        out = defaultdict(list)
        for source, target, u, v in self.arrows:
            out[source].append((target, u, v))
        for source in range(size):
            total: set = set()
            for middle, u, v in out[source]:
                for target, onward_u, onward_v in out[middle]:
                    if not (u + onward_u and v + onward_v):
                        total ^= {(target, u + onward_u, v + onward_v)}
            if total:
                raise TetradromeError(f"d^2 is not zero at generator {source}.")

    @classmethod
    def from_type_d(cls, typed: TypeD) -> KnotComplex:
        """The complex of the curved type D structure above the global minimum."""
        algebra = typed.algebra
        if not algebra.curved or algebra.strands != 2 or algebra.occupied != 1:
            raise TetradromeError(f"the terminal module pairs with the curved B(2, 1), not "
                                  f"{algebra}.")
        (up,) = algebra.upward
        gradings = []
        for generator in range(len(typed)):
            if typed.idempotents[generator] != _ONE:
                raise TetradromeError("a generator above the global minimum is off the "
                                      "idempotent {1}.")
            if typed.alexander[generator] % 2:
                raise TetradromeError("a generator above the global minimum has a "
                                      "half-integral Alexander grading.")
            gradings.append((typed.maslov[generator], typed.alexander[generator] // 2))
        arrows = []
        for source in range(len(typed)):
            for target, coefficient in typed.arrows[source].items():
                if len(coefficient) != 1:
                    raise TetradromeError(f"the arrow {source} -> {target} carries "
                                          f"{len(coefficient)} terms.")
                (element,) = coefficient
                doubled = unpack(element.weights, 2)
                arrows.append((source, target, doubled[up] // 2, doubled[1 - up] // 2))
        return cls(tuple(gradings), tuple(arrows))

    @property
    def horizontal(self) -> tuple:
        """The U arrows ``(source, target, u)``."""
        return tuple((source, target, u) for source, target, u, _ in self.arrows if u)

    @property
    def vertical(self) -> tuple:
        """The V arrows ``(source, target, v)``."""
        return tuple((source, target, v) for source, target, _, v in self.arrows if v)

    def dual(self) -> KnotComplex:
        """Hom(C, R): the complex of the mirror (Proposition 10.3)."""
        return KnotComplex(tuple((-maslov, -alexander) for maslov, alexander in self.gradings),
                           tuple((target, source, u, v)
                                 for source, target, u, v in self.arrows))

    def hat_ranks(self) -> dict:
        """HFK-hat ``{(maslov, alexander): rank}``: the homology at U = V = 0."""
        members: dict = defaultdict(lambda: defaultdict(list))
        place = {}
        for generator, (maslov, alexander) in enumerate(self.gradings):
            place[generator] = len(members[alexander][-maslov])
            members[alexander][-maslov].append(generator)
        unit = defaultdict(list)
        for source, target, u, v in self.arrows:
            if not u and not v:
                unit[source].append(target)
        ranks: dict = defaultdict(int)
        for alexander, by_degree in members.items():
            maps = {}
            for degree, sources in by_degree.items():
                columns = [[place[target] for target in unit[source]] for source in sources]
                if any(columns):
                    maps[degree] = columns
            dimensions = {degree: len(sources) for degree, sources in by_degree.items()}
            for degree, rank in homology(GradedComplex(dimensions, maps)).items():
                ranks[(-degree, alexander)] += rank
        return dict(ranks)

    @cached_property
    def _vertical(self) -> _Localized:
        return _Localized(self, lambda u, v: not u, lambda maslov, alexander: maslov)

    @cached_property
    def _horizontal(self) -> _Localized:
        return _Localized(self, lambda u, v: not v,
                          lambda maslov, alexander: maslov - 2 * alexander)

    def _levels(self) -> range:
        alexanders = [alexander for _, alexander in self.gradings]
        return range(min(alexanders), max(alexanders) + 1)

    def _tau_pair(self) -> tuple:
        """tau from the vertical complex and the paper's tau from the horizontal one."""
        vertical, horizontal = self._vertical, self._horizontal
        least = _first(self._levels(), lambda level: _reaches(
            [g for g in vertical.index if self.gradings[g][1] <= level],
            vertical.out.__getitem__, vertical.rows, _itself, vertical))
        largest = _first(reversed(self._levels()), lambda level: _reaches(
            [g for g in horizontal.index if self.gradings[g][1] >= level],
            horizontal.out.__getitem__, horizontal.rows, _itself, horizontal))
        return least, -largest

    def _reaches_at(self, level: int, from_below: bool) -> bool:
        """Whether H_0 of A-hat at ``level`` maps onto HF-hat: into the vertical complex by
        dropping the generators above ``level``, or into the horizontal one by dropping those
        below."""
        out = defaultdict(list)
        for source, target, u, v in self.arrows:
            alexander = self.gradings[source][1]
            if (not u or alexander >= level) and (not v or alexander <= level):
                out[source].append(target)
        degree = [maslov - 2 * max(0, alexander - level) for maslov, alexander in self.gradings]
        top = 0 if from_below else 2 * level
        sources = [g for g, value in enumerate(degree) if value == top]
        rows = {g: position for position, g in enumerate(
            g for g, value in enumerate(degree) if value == top - 1)}
        if from_below:
            return _reaches(sources, out.__getitem__, rows,
                            lambda g: g if self.gradings[g][1] <= level else None,
                            self._vertical)
        return _reaches(sources, out.__getitem__, rows,
                        lambda g: g if self.gradings[g][1] >= level else None, self._horizontal)

    def _nu_pair(self) -> tuple:
        """nu from the vertical complex and the paper's nu from the horizontal one."""
        least = _first(self._levels(), lambda level: self._reaches_at(level, True))
        largest = _first(reversed(self._levels()), lambda level: self._reaches_at(level, False))
        return least, -largest

    def tau(self) -> int:
        """Ozsvath and Szabo's tau."""
        value, paper = self._tau_pair()
        if value != paper:
            raise TetradromeError(f"tau is {value} from the vertical complex and {paper} from "
                                  f"the horizontal one.")
        return value

    def nu(self) -> int:
        """Ozsvath and Szabo's nu."""
        value, paper = self._nu_pair()
        if value != paper:
            raise TetradromeError(f"nu is {value} from the vertical complex and {paper} from "
                                  f"the horizontal one.")
        tau = self.tau()
        if not tau <= value <= tau + 1:
            raise TetradromeError(f"nu = {value} is not tau = {tau} or tau + 1.")
        return value

    def epsilon(self) -> int:
        """Hom's epsilon, by (11.2)."""
        mirror = self.dual()
        tau, mirror_tau = self.tau(), mirror.tau()
        if mirror_tau != -tau:
            raise TetradromeError(f"the dual complex has tau {mirror_tau}, not {-tau}.")
        value = (tau - self.nu()) - (mirror_tau - mirror.nu())
        if value not in (-1, 0, 1):
            raise TetradromeError(f"epsilon = {value} is not -1, 0 or 1.")
        return value
