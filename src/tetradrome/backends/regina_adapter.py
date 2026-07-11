# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Regina as a SPEC 12.1 validator: determinant, Alexander, and Jones cross-checks.

Self-contained by design: the comparison layer (``scripts/comparison/adapters.py``)
keeps its own measurement-oriented Regina path, and the two deliberately do not share
code -- the operator's call, made when this validator was wired. Do not unify them.

Conventions, verified empirically on chiral and amphichiral knots (3_1, 4_1, 5_2,
8_19, 10_124) before wiring:

- Jones: regina returns a Laurent polynomial in x = t^(1/2) whose exponents are all
  even for a knot; halving them gives native's t variable DIRECTLY (no t <-> t^-1
  flip). Odd exponents fail loud.
- Alexander: regina's polynomial, canonicalized up to the defining +/- t^k unit,
  equals native's canonical form exactly.
- Determinant: |Delta(-1)| evaluated from regina's Alexander polynomial -- the
  defining identity computed through an independent code path (regina's Link API
  exposes no direct determinant or signature).

Any future disagreement raises through the validation machinery; nothing is silently
re-normalized beyond the fixed transforms above. Cross-package imports are deferred to
call time (the registry imports this module while ``invariants`` may still be
mid-initialization).
"""
from __future__ import annotations

_COVERED = {"determinant", "alexander_polynomial", "jones_polynomial"}


def _split_laurent_terms(text: str) -> list[str]:
    """Split a single-variable Laurent polynomial into signed terms, treating ``+``/``-``
    as separators except immediately after ``^`` (an exponent sign)."""
    text = text.replace(" ", "")
    terms: list[str] = []
    current = ""
    for index, character in enumerate(text):
        if character in "+-" and index > 0 and text[index - 1] != "^":
            terms.append(current)
            current = character
        else:
            current += character
    if current:
        terms.append(current)
    return terms


def _parse_laurent(text: str, variable: str) -> dict[int, int]:
    """``[sign][coeff]var[^exp]`` terms -> {exponent: coefficient}, zeros dropped.

    Handles signs, negative exponents, an implicit exponent 1, a bare constant term,
    and ``*``-joined factors. Anything else fails loud, as does an empty polynomial
    (no invariant this validator reads is ever zero).
    """
    poly: dict[int, int] = {}
    for term in _split_laurent_terms(text):
        if not term or term in "+-":
            continue
        term = term.replace("*", "")
        sign = 1
        if term[:1] == "+":
            term = term[1:]
        elif term[:1] == "-":
            sign, term = -1, term[1:]
        if variable in term:
            left, _, right = term.partition(variable)
            coefficient = int(left) if left else 1
            exponent = int(right[1:]) if right.startswith("^") else 1
        else:
            exponent, coefficient = 0, int(term)
        poly[exponent] = poly.get(exponent, 0) + sign * coefficient
    poly = {exponent: c for exponent, c in poly.items() if c}
    if not poly:
        raise ValueError(f"empty or unparseable Laurent polynomial: {text!r}")
    return poly


def _ascending_coefficients(poly: dict[int, int]) -> tuple[int, list[int]]:
    """{exponent: coefficient} -> (lowest exponent, dense ascending coefficient list)."""
    low, high = min(poly), max(poly)
    return low, [poly.get(exponent, 0) for exponent in range(low, high + 1)]


class ReginaValidator:
    """Read-only cross-check against Regina's Link engine (SPEC 12.1, ADR 0006)."""

    name = "regina"
    covered_invariants = _COVERED

    def is_available(self) -> bool:
        try:
            import regina  # noqa: F401
        except ImportError:
            return False
        return True

    def version_info(self) -> dict:
        """Pip metadata, mirroring the install_oracles.sh / comparison-layer derivation."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            return {"regina": version("regina")}
        except PackageNotFoundError:
            return {"regina": "absent"}

    def known_value(self, knot, invariant: str):
        """Regina's value under the canonical name and convention, or None when regina
        cannot check this input (uncovered invariant, or a knot with no PD)."""
        if invariant not in _COVERED or not knot.pd_code:
            return None
        import regina

        link = regina.Link.fromPD([list(crossing) for crossing in knot.pd_code])
        if invariant == "jones_polynomial":
            from ..invariants import jones

            x_poly = _parse_laurent(str(link.jones()), "x")
            if any(exponent % 2 for exponent in x_poly):
                raise ValueError(
                    f"unexpected regina Jones (x = t^1/2 must have even exponents): "
                    f"{str(link.jones())!r}"
                )
            t_poly = {exponent // 2: c for exponent, c in x_poly.items()}
            return jones.canonical_laurent(*_ascending_coefficients(t_poly))

        alexander = _parse_laurent(str(link.alexander()), "x")
        if invariant == "determinant":
            return abs(sum(c * (-1) ** exponent for exponent, c in alexander.items()))
        from ..invariants import seifert

        _, coefficients = _ascending_coefficients(alexander)
        return seifert.canonical_alexander(coefficients)
