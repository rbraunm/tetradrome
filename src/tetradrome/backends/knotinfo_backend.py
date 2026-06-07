"""KnotInfo backend.

Read access to the offline KnotInfo table (`database_knotinfo`). Serves two roles:
the source of structural data for tabulated knots (PD code, braid word) and the
known-answer oracle for validation. It is optional: a missing install raises
BackendUnavailable, never a silent fallback (decisions/0004).

KnotInfo names use the forms `3_1` (Rolfsen, <= 10 crossings) and `11n_34`
(Hoste-Thistlethwaite, >= 11). `normalize_name` maps common spellings (a leading
`K`, or a missing underscore as Spherogram emits) onto these.
"""
from __future__ import annotations

import ast
import re

from ..errors import BackendUnavailable, UnknownKnot

_TABLE: list[dict] | None = None
_BY_NAME: dict[str, dict] | None = None

# Map our canonical invariant names onto KnotInfo columns (conventions.md, SPEC 12.4).
_ORACLE_COLUMN = {
    "determinant": "determinant",
    "signature": "signature",
}

# e.g. "11n34" -> ("11n", "34");  "10_124" already has the underscore.
_ALPHA_FORM = re.compile(r"^(\d+[a-zA-Z])(\d+)$")


def _blank(raw) -> bool:
    return raw is None or str(raw).strip() in ("", "does not exist")


def _khovanov_integral_vector(name: str):
    """KnotInfo's unreduced integral Khovanov as a list of [torsion, mult, i, j], or
    None if the knot has no stored vector."""
    raw = lookup(name).get("khovanov_unreduced_integral_vector")
    if _blank(raw):
        return None
    return ast.literal_eval(str(raw))


def _khovanov_free_ranks(name: str):
    """Rational (torsion-free) unreduced Khovanov: dim_Q Kh^{i,j} = the free summands."""
    vec = _khovanov_integral_vector(name)
    if vec is None:
        return None
    out: dict[tuple[int, int], int] = {}
    for torsion, mult, i, j in vec:
        if torsion == 0:
            out[(i, j)] = out.get((i, j), 0) + mult
    return out


def _khovanov_mod2(name: str):
    """Mod-2 unreduced Khovanov by universal coefficients (cohomological): dim_F2 Kh^{i,j}
    = free rank + 2-torsion at (i,j) + 2-torsion at (i+1,j)."""
    vec = _khovanov_integral_vector(name)
    if vec is None:
        return None
    free: dict[tuple[int, int], int] = {}
    tor2: dict[tuple[int, int], int] = {}
    for torsion, mult, i, j in vec:
        if torsion == 0:
            free[(i, j)] = free.get((i, j), 0) + mult
        elif torsion % 2 == 0:
            tor2[(i, j)] = tor2.get((i, j), 0) + mult
    out: dict[tuple[int, int], int] = {}
    for (i, j) in set(free) | set(tor2) | {(i - 1, j) for (i, j) in tor2}:
        d = free.get((i, j), 0) + tor2.get((i, j), 0) + tor2.get((i + 1, j), 0)
        if d:
            out[(i, j)] = d
    return out


def _mirror_bigraded(table):
    """KnotInfo tabulates Khovanov (and s) in the opposite chirality from its stored PD
    (verified across the small-knot table). Our value is the correct invariant of the
    given diagram, so the oracle is mirrored to match: (i, j) -> (-i, -j)."""
    return {(-i, -j): d for (i, j), d in table.items()}


def _load():
    global _TABLE, _BY_NAME
    if _BY_NAME is not None:
        return
    try:
        import database_knotinfo
    except ImportError as exc:
        raise BackendUnavailable(
            "KnotInfo backend needs 'database_knotinfo' (a core dependency; "
            "reinstall with 'pip install -e .' or 'pip install database_knotinfo')."
        ) from exc
    rows = [
        r
        for r in database_knotinfo.link_list()
        if isinstance(r, dict) and r.get("name") and r.get("name") != "Name"
    ]
    _TABLE = rows
    _BY_NAME = {r["name"]: r for r in rows}


def normalize_name(name: str) -> str:
    """Map a knot name onto KnotInfo's spelling. Does not verify existence."""
    n = name.strip()
    if n.startswith("K"):
        n = n[1:]
    if "_" in n:
        return n
    m = _ALPHA_FORM.match(n)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return n


def lookup(name: str) -> dict:
    """Return the KnotInfo row for a knot, or raise UnknownKnot."""
    _load()
    key = normalize_name(name)
    assert _BY_NAME is not None
    row = _BY_NAME.get(key)
    if row is None:
        raise UnknownKnot(f"{name!r} (normalized {key!r}) is not in KnotInfo.")
    return row


def pd_notation(name: str) -> list:
    """Parse a knot's PD code from KnotInfo into a nested list of ints."""
    raw = lookup(name).get("pd_notation")
    if not raw:
        raise UnknownKnot(f"{name!r} has no pd_notation in KnotInfo.")
    return ast.literal_eval(raw)


def grid_notation(name: str) -> list[list[int]]:
    """Parse a knot's grid diagram from KnotInfo into a list of [row, col] markers
    (1-based, 2n of them). Like the PD code and braid word, this is a *presentation* of
    the knot -- input to the grid-homology engine -- not a precomputed answer.
    """
    raw = lookup(name).get("grid_notation")
    if not raw:
        raise UnknownKnot(f"{name!r} has no grid_notation in KnotInfo.")
    return ast.literal_eval(raw)


def hfk_ranks(name: str) -> dict[tuple[int, int], int]:
    """Parse KnotInfo's HFK-hat into ``{(Maslov, Alexander): rank}`` (the oracle for the
    grid-homology engine). The stored vector lists ``rank, Alexander, Maslov`` triples."""
    raw = lookup(name).get("hfk_polynomial_vector")
    if not raw:
        raise UnknownKnot(f"{name!r} has no hfk_polynomial_vector in KnotInfo.")
    out: dict[tuple[int, int], int] = {}
    for triple in str(raw).strip().strip("[]").split(";"):
        rank, alexander_grading, maslov_grading = (int(v) for v in triple.split(","))
        out[(maslov_grading, alexander_grading)] = rank
    return out


def tau_invariant(name: str) -> int:
    """KnotInfo's Ozsvath-Szabo tau invariant (the oracle for the grid tau)."""
    raw = lookup(name).get("ozsvath_szabo_tau_invariant")
    if raw is None or str(raw).strip() == "":
        raise UnknownKnot(f"{name!r} has no tau invariant in KnotInfo.")
    return int(str(raw).strip())


def braid_word(name: str) -> list[int]:
    """Parse a knot's braid word from KnotInfo into a list of nonzero ints.

    The braid word is a *presentation* of the knot (input to native Seifert-matrix
    computation), not a precomputed answer.
    """
    raw = lookup(name).get("braid_notation")
    if not raw:
        raise UnknownKnot(f"{name!r} has no braid_notation in KnotInfo.")
    s = str(raw).strip().replace("{", "[").replace("}", "]").replace(";", ",")
    value = ast.literal_eval(s)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(isinstance(v, list) for v in value):
        raise UnknownKnot(f"{name!r} braid_notation is not a flat word: {raw!r}")
    return [int(v) for v in value]


def known_answer(name: str, invariant: str):
    """KnotInfo's stored value for `invariant`, or None if not available.

    Integers (determinant, signature) come back as int; the Alexander polynomial comes
    back as ascending integer coefficients (a tuple), left in KnotInfo's raw form for
    the invariants layer to canonicalize. None means the oracle has no value
    (blank/sentinel) -- it is never coerced to a default (decisions/0004).
    """
    if invariant == "alexander_polynomial":
        raw = lookup(name).get("alexander_polynomial_vector")
        if raw is None or str(raw).strip() in ("", "does not exist"):
            return None
        vec = ast.literal_eval(str(raw))
        return tuple(int(c) for c in vec[2:])  # vec = [low_exp, high_exp, c_low, ...]

    if invariant == "jones_polynomial":
        raw = lookup(name).get("jones_polynomial_vector")
        if raw is None or str(raw).strip() in ("", "does not exist"):
            return None
        vec = ast.literal_eval(str(raw))  # [low_exp, high_exp, c_low, ...]
        return (int(vec[0]), tuple(int(c) for c in vec[2:]))

    # Homological invariants. KnotInfo's Khovanov/s columns use the mirror chirality of
    # its PD, so the oracle is mirrored / sign-flipped to match our value (Phase 2c/3c).
    if invariant == "khovanov_homology":
        table = _khovanov_mod2(name)
        return None if table is None else _mirror_bigraded(table)

    if invariant == "rational_khovanov_homology":
        table = _khovanov_free_ranks(name)
        return None if table is None else _mirror_bigraded(table)

    if invariant == "rasmussen_s":
        raw = lookup(name).get("rasmussen_invariant")
        if _blank(raw):
            return None
        return -int(str(raw).strip())  # s(mirror) = -s

    column = _ORACLE_COLUMN.get(invariant)
    if column is None:
        return None
    raw = lookup(name).get(column)
    if raw is None or str(raw).strip() == "" or str(raw).strip() == "does not exist":
        return None
    return int(str(raw).strip())
