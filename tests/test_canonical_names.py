# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The canonical-name record and the code must agree.

Decision 0001 is the only list of canonical invariant names, and SPEC 12.4 carries the
same set with each backend's spelling alongside. Both drifted from ``compute()`` with
nothing noticing: the record named ``rasmussen_invariant`` while ``compute()`` accepted
only ``rasmussen_s``, and ``rational_khovanov_homology`` was in neither. These tests
read both documents and pin them against the code and against each other.
"""
import pathlib
import re

from tetradrome.invariants.compute import _supported

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADR_0001 = ROOT / "roadmap" / "decisions" / "0001-canonical-invariant-names.md"
SPEC = ROOT / "SPEC.md"


def _canonical_column(path: pathlib.Path, header_start: str) -> list[str]:
    """The backticked name opening each row of the markdown table whose header row
    starts with ``header_start``. Fails loud on a missing table or an unparseable row
    rather than quietly returning a short list."""
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(header_start)]
    assert len(starts) == 1, f"expected one table headed {header_start!r} in {path.name}"
    names = []
    for line in lines[starts[0] + 2:]:
        if not line.startswith("|"):
            break
        match = re.match(r"\|\s*`([a-z0-9_]+)`", line)
        assert match, f"unparseable canonical row in {path.name}: {line!r}"
        names.append(match.group(1))
    assert names, f"empty canonical table in {path.name}"
    return names


def _adr_names() -> list[str]:
    return _canonical_column(ADR_0001, "| Canonical name |")


def _spec_names() -> list[str]:
    return _canonical_column(SPEC, "| Tetradrome (canonical) |")


def test_every_computed_invariant_is_in_the_canonical_record():
    missing = sorted(set(_supported()) - set(_adr_names()))
    assert missing == []


def test_spec_table_carries_exactly_the_canonical_set():
    assert sorted(_spec_names()) == sorted(_adr_names())


def test_canonical_tables_have_no_duplicate_rows():
    for names in (_adr_names(), _spec_names()):
        assert len(names) == len(set(names))
