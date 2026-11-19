# SPDX-License-Identifier: Apache-2.0

"""The local crossing bimodule P over B(2) (arXiv:1603.06559, sections 5.1 and 5.3).

Its operations are written on types: an element of B(2) is named by ``Word(c1, c2, letters, u1,
u2)``, the monomial U1^u1 U2^u2 times one of the words 1, L1, R1, L2, R2, L1L2, R2R1, times C1 and
C2 where the flags are set. Generators are named N, S, W, E. ``local_delta(kind, inputs)``
returns the terms ``(Word, kind)`` of delta^1_l on a generator of that kind, l - 1 being the
number of inputs; the global bimodule decides which of them exist at given idempotents.

delta^1_2 is the closure of the sixteen terms of diagram (5.4) under the rules after it, with
the C's carried by (5.12) on the actions among N, W, E and from S to S, plus the actions from S
of section 5.3 listed literally and extended by U1 U2. delta^1_3 starts only at S: the list of
section 5.1, extended by (5.6) and by the C rule of section 5.3.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple


class Word(NamedTuple):
    c1: int
    c2: int
    letters: str
    u1: int
    u2: int


def _word(letters: str, u1: int = 0, u2: int = 0, c1: int = 0, c2: int = 0) -> Word:
    return Word(c1, c2, letters, u1, u2)


def _carried_across(word: Word, c1: int, c2: int) -> Word | None:
    """``word`` times C2^c1 C1^c2: an input's C on one strand leaves on the other."""
    if word.c1 and c2 or word.c2 and c1:
        return None
    return word._replace(c1=word.c1 | c2, c2=word.c2 | c1)


def _times_u1u2(word: Word, power: int) -> Word:
    return word._replace(u1=word.u1 + power, u2=word.u2 + power)


def _delta_two_plain(kind: str, word: Word) -> list:
    """delta^1_2 on a C-free input, in closed form from the terms of (5.4) and its rules."""
    letters, u1, u2 = word.letters, word.u1, word.u2
    if kind == "N":
        base = {"1": [(_word("1"), "N")],
                "L1L2": [(_word("L1L2"), "N")],
                "R2R1": [(_word("R2R1"), "N")],
                "L1": [(_word("1", 0, 1), "W"), (_word("L1L2"), "E")],
                "R2": [(_word("R2R1"), "W"), (_word("1", 1, 0), "E")]}.get(letters, [])
        return [(out._replace(u1=out.u1 + u2, u2=out.u2 + u1), target) for out, target in base]
    if kind == "W":
        if u1 < u2:
            return []
        gamma, common = u1 - u2, u2
        if letters == "1":
            base = [(_word("1", 0, gamma), "W")]
            if gamma >= 1:
                base.append((_word("L1L2", 0, gamma - 1), "E"))
        elif letters == "R1":
            base = [(_word("1", 0, gamma), "N")]
        elif letters == "L2" and gamma >= 1:
            base = [(_word("L1L2", 0, gamma - 1), "N")]
        else:
            base = []
        return [(_times_u1u2(out, common), target) for out, target in base]
    if kind == "E":
        if u2 < u1:
            return []
        gamma, common = u2 - u1, u1
        if letters == "1":
            base = [(_word("1", gamma, 0), "E")]
            if gamma >= 1:
                base.append((_word("R2R1", gamma - 1, 0), "W"))
        elif letters == "L2":
            base = [(_word("1", gamma, 0), "N")]
        elif letters == "R1" and gamma >= 1:
            base = [(_word("R2R1", gamma - 1, 0), "N")]
        else:
            base = []
        return [(_times_u1u2(out, common), target) for out, target in base]
    if letters == "1" and u1 == u2:
        return [(_word("1", u1, u2), "S")]
    return []


# The actions from S of section 5.3, keyed by the input with its powers of U1 U2 removed.
_FROM_S_WITH_C = {
    _word("1", c2=1): [(_word("R1"), "W")],
    _word("1", c1=1): [(_word("L2"), "E")],
    _word("1", c1=1, c2=1): [(_word("R1", c2=1), "W"), (_word("L2", c1=1), "E")],
    _word("1", 1, 0, c2=1): [(_word("L2", 1, 0), "E")],
    _word("1", 1, 0, c1=1, c2=1): [(_word("L2", 1, 0, c2=1), "E")],
    _word("1", 0, 1, c1=1): [(_word("R1", 0, 1), "W")],
    _word("1", 0, 1, c1=1, c2=1): [(_word("R1", 0, 1, c1=1), "W")],
    _word("R1", c2=1): [(_word("R1"), "N")],
    _word("R1", c1=1, c2=1): [(_word("R1", c2=1), "N")],
    _word("L2", c1=1): [(_word("L2"), "N")],
    _word("L2", c1=1, c2=1): [(_word("L2", c1=1), "N")],
    _word("R1", 0, 1, c1=1): [(_word("R1", 0, 1), "N")],
    _word("R1", 0, 1, c1=1, c2=1): [(_word("R1", 0, 1, c1=1), "N")],
    _word("L2", 1, 0, c2=1): [(_word("L2", 1, 0), "N")],
    _word("L2", 1, 0, c1=1, c2=1): [(_word("L2", 1, 0, c2=1), "N")],
}


def _delta_two(kind: str, word: Word) -> list:
    plain = word._replace(c1=0, c2=0)
    terms = []
    for out, target in _delta_two_plain(kind, plain):
        carried = _carried_across(out, word.c1, word.c2)
        if carried is not None:
            terms.append((carried, target))
    if kind == "S":
        common = min(word.u1, word.u2)
        stripped = _times_u1u2(word, -common)
        terms.extend((_times_u1u2(out, common), target)
                     for out, target in _FROM_S_WITH_C.get(stripped, []))
    return terms


# The list of section 5.1 for delta^1_3(S, a1, a2), on inputs free of C's and of U1 U2. An
# exponent is (n, t, constant), the value n * N + t * T + constant for the list's n and t.
def _n(constant: int = 0) -> tuple:
    return (1, 0, constant)


def _t(constant: int = 0) -> tuple:
    return (0, 1, constant)


_ZERO = (0, 0, 0)


def _below(n, t):
    return 0 <= n < t


def _above(n, t):
    return 1 <= t <= n


def _t_below(n, t):
    return 0 <= t < n


def _n_above(n, t):
    return 1 <= n <= t


_DELTA_THREE = [
    # (output word, output exponents, target, condition on (n, t), [(a1, a2), ...])
    ("R1", (_t(), _ZERO), "E", lambda n, t: n == 0 and t >= 0,
     [(("R1", _ZERO, _ZERO), ("R2", _ZERO, _t()))]),
    ("L2", (_t(), _n()), "E", _below,
     [(("1", _n(1), _ZERO), ("1", _ZERO, _t())),
      (("R1", _n(), _ZERO), ("L1", _ZERO, _t())),
      (("L2", _n(1), _ZERO), ("R2", _ZERO, _t(-1)))]),
    ("L2", (_t(), _n()), "E", _above,
     [(("1", _ZERO, _t()), ("1", _n(1), _ZERO)),
      (("R1", _ZERO, _t()), ("L1", _n(), _ZERO)),
      (("L2", _ZERO, _t(-1)), ("R2", _n(1), _ZERO))]),
    ("L2", (_ZERO, _n()), "W", lambda n, t: t == 0 and n >= 0,
     [(("L2", _ZERO, _ZERO), ("L1", _n(), _ZERO))]),
    ("R1", (_t(), _n()), "W", _t_below,
     [(("1", _ZERO, _t(1)), ("1", _n(), _ZERO)),
      (("L2", _ZERO, _t()), ("R2", _n(), _ZERO)),
      (("R1", _ZERO, _t(1)), ("L1", _n(-1), _ZERO))]),
    ("R1", (_t(), _n()), "W", _n_above,
     [(("1", _n(), _ZERO), ("1", _ZERO, _t(1))),
      (("L2", _n(), _ZERO), ("R2", _ZERO, _t())),
      (("R1", _n(-1), _ZERO), ("L1", _ZERO, _t(1)))]),
    ("L2", (_t(), _n()), "N", _below,
     [(("1", _n(1), _ZERO), ("L2", _ZERO, _t())),
      (("R1", _n(), _ZERO), ("L1L2", _ZERO, _t())),
      (("L2", _n(1), _ZERO), ("1", _ZERO, _t()))]),
    ("L2", (_t(), _n()), "N", _above,
     [(("L2", _ZERO, _t()), ("1", _n(1), _ZERO)),
      (("1", _ZERO, _t()), ("L2", _n(1), _ZERO)),
      (("R1", _ZERO, _t()), ("L1L2", _n(), _ZERO))]),
    ("L2", (_ZERO, _n()), "N", lambda n, t: t == 0 and n >= 0,
     [(("L2", _ZERO, _ZERO), ("1", _n(1), _ZERO))]),
    ("R1", (_t(), _n()), "N", _t_below,
     [(("1", _ZERO, _t(1)), ("R1", _n(), _ZERO)),
      (("L2", _ZERO, _t()), ("R2R1", _n(), _ZERO)),
      (("R1", _ZERO, _t(1)), ("1", _n(), _ZERO))]),
    ("R1", (_t(), _n()), "N", _n_above,
     [(("R1", _n(), _ZERO), ("1", _ZERO, _t(1))),
      (("1", _n(), _ZERO), ("R1", _ZERO, _t(1))),
      (("L2", _n(), _ZERO), ("R2R1", _ZERO, _t()))]),
    ("R1", (_t(), _ZERO), "N", lambda n, t: n == 0 and t >= 0,
     [(("R1", _ZERO, _ZERO), ("1", _ZERO, _t(1)))]),
]


def _solve(pattern: list, values: list) -> tuple | None:
    """The ``(n, t)`` that make the exponent patterns equal the values, unused ones zero."""
    unknowns = [None, None]
    for (n_coefficient, t_coefficient, constant), value in zip(pattern, values):
        if n_coefficient:
            candidate, slot = value - constant, 0
        elif t_coefficient:
            candidate, slot = value - constant, 1
        else:
            if value != constant:
                return None
            continue
        if unknowns[slot] is not None and unknowns[slot] != candidate:
            return None
        unknowns[slot] = candidate
    return tuple(0 if unknown is None else unknown for unknown in unknowns)


def _evaluate(exponent: tuple, n: int, t: int) -> int:
    return exponent[0] * n + exponent[1] * t + exponent[2]


def _delta_three_plain(first: Word, second: Word) -> list:
    terms = []
    for letters, (out_u1, out_u2), target, condition, cases in _DELTA_THREE:
        for (letters_one, u1_one, u2_one), (letters_two, u1_two, u2_two) in cases:
            if letters_one != first.letters or letters_two != second.letters:
                continue
            solution = _solve([u1_one, u2_one, u1_two, u2_two],
                              [first.u1, first.u2, second.u1, second.u2])
            if solution is None or not condition(*solution):
                continue
            terms.append((_word(letters, _evaluate(out_u1, *solution),
                                _evaluate(out_u2, *solution)), target))
    return terms


def _delta_three(kind: str, first: Word, second: Word) -> list:
    if kind != "S":
        return []
    if first.c1 and second.c1 or first.c2 and second.c2:
        return []
    common = min(first.u1, first.u2) + min(second.u1, second.u2)
    plain_first = _times_u1u2(first._replace(c1=0, c2=0), -min(first.u1, first.u2))
    plain_second = _times_u1u2(second._replace(c1=0, c2=0), -min(second.u1, second.u2))
    c1, c2 = first.c1 | second.c1, first.c2 | second.c2
    return [(_times_u1u2(_carried_across(out, c1, c2), common), target)
            for out, target in _delta_three_plain(plain_first, plain_second)]


@lru_cache(maxsize=None)
def local_delta(kind: str, inputs: tuple) -> frozenset:
    """The terms ``(Word, kind)`` of delta^1 on a generator of kind ``kind`` with these inputs."""
    if not inputs:
        terms = {"W": [(_word("L1"), "S")], "E": [(_word("R2"), "S")]}.get(kind, [])
    elif len(inputs) == 1:
        terms = _delta_two(kind, inputs[0])
    elif len(inputs) == 2:
        terms = _delta_three(kind, *inputs)
    else:
        terms = []
    total: set = set()
    for term in terms:
        total ^= {term}
    return frozenset(total)
