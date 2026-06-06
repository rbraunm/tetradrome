"""KnotInfo backend.

Read access to the offline KnotInfo table (`database_knotinfo`). Serves two roles:
the source of structural data for tabulated knots (PD code, Seifert matrix) and the
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


def _load():
    global _TABLE, _BY_NAME
    if _BY_NAME is not None:
        return
    try:
        import database_knotinfo
    except ImportError as exc:
        raise BackendUnavailable(
            "KnotInfo backend needs 'database_knotinfo' (pip install tetradrome[knotinfo])."
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


def seifert_matrix(name: str) -> list[list[int]]:
    """Parse a knot's Seifert matrix from KnotInfo into a list of int rows."""
    raw = lookup(name).get("seifert_matrix")
    if not raw:
        raise UnknownKnot(f"{name!r} has no seifert_matrix in KnotInfo.")
    return [list(row) for row in ast.literal_eval(raw)]


def known_answer(name: str, invariant: str):
    """KnotInfo's stored integer value for `invariant`, or None if not available.

    None means the oracle has no value (blank/sentinel) -- it is never coerced to a
    default (decisions/0004).
    """
    column = _ORACLE_COLUMN.get(invariant)
    if column is None:
        return None
    raw = lookup(name).get(column)
    if raw is None or str(raw).strip() == "" or str(raw).strip() == "does not exist":
        return None
    return int(str(raw).strip())
