# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The canonical-name table and the code must agree.

SPEC 7 is the only list of canonical invariant names, with each tool's spelling alongside and
a column marking the ones computed natively. These tests read it and pin it against
``compute()``.
"""
import pathlib
import re

from tetradrome.invariants.compute import _supported

SPEC = pathlib.Path(__file__).resolve().parents[1] / "SPEC.md"
_HEADER = "| Tetradrome (canonical) | Native |"
_ROW = re.compile(r"\|\s*`([a-z0-9_]+)`[^|]*\|\s*([^|]*?)\s*\|")
_NOT_NATIVE = "\u2014"


def _canonical_rows() -> list[tuple[str, bool]]:
    """(name, computed natively) for each row of the SPEC 7 table. Fails loud on a missing
    table or an unparseable row rather than quietly returning a short list."""
    lines = SPEC.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(_HEADER)]
    assert len(starts) == 1, f"expected one table headed {_HEADER!r} in SPEC.md"
    rows = []
    for line in lines[starts[0] + 2:]:
        if not line.startswith("|"):
            break
        match = _ROW.match(line)
        assert match, f"unparseable canonical row in SPEC.md: {line!r}"
        assert match.group(2) in ("yes", _NOT_NATIVE), f"unreadable Native cell: {line!r}"
        rows.append((match.group(1), match.group(2) == "yes"))
    assert rows, "empty canonical table in SPEC.md"
    return rows


def test_native_column_is_exactly_what_compute_accepts():
    assert sorted(name for name, native in _canonical_rows() if native) == sorted(_supported())


def test_canonical_table_has_no_duplicate_rows():
    names = [name for name, _ in _canonical_rows()]
    assert len(names) == len(set(names))
