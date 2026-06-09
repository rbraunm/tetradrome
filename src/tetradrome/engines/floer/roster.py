"""The KnotInfo validation roster for the grid-Floer engine.

A knot is in the roster when KnotInfo tabulates everything the engine validates against --
HFK-hat, the Ozsvath-Szabo tau, and the three-genus -- and its grid number is within a
tractable bound. Derived live from the table so it tracks the installed KnotInfo rather than
freezing a hand-maintained list. The n <= 10 acceptance ceiling is the brute floor: the grid
complex at n = 11 overruns 200 GiB (decisions/0011; roadmap/design/floer-phase-6-plan.md).
"""
from __future__ import annotations

import ast

from ...backends import knotinfo_backend as ki

_REQUIRED = (
    "grid_notation",
    "hfk_polynomial_vector",
    "ozsvath_szabo_tau_invariant",
    "three_genus",
)


def _blank(value) -> bool:
    return value is None or str(value).strip() in ("", "does not exist")


def floer_roster(max_n: int) -> list[tuple[str, int]]:
    """KnotInfo knots with HFK, tau, and three-genus tabulated and grid number ``<= max_n``,
    as sorted ``(name, n)``. The grid number is the marker count of the stored grid diagram."""
    out: list[tuple[str, int]] = []
    for row in ki.rows():
        if any(_blank(row.get(col)) for col in _REQUIRED):
            continue
        try:
            n = len(ast.literal_eval(row["grid_notation"])) // 2
        except (ValueError, SyntaxError):
            continue
        if n <= max_n:
            out.append((row["name"], n))
    out.sort(key=lambda item: (item[1], item[0]))
    return out
