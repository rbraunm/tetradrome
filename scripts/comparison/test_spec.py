# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Consistency checks on the comparison artifact's hand-authored catalog (spec.py).

spec.py is reviewed prose, not generated, so nothing kept its status flags honest:
``lee_homology`` sat at ``done`` -- defined as implemented natively and validated --
while having no native compute, so the artifact rendered a done row as "not
implemented". These pin the catalog against the code it describes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec  # noqa: E402

from tetradrome.invariants.compute import _supported  # noqa: E402


def test_every_done_row_has_a_native_compute():
    assert [row.name for row in spec.INVARIANTS if row.status == "done" and row.tetra is None] == []


def test_benchmark_rows_cover_exactly_the_native_invariants():
    """Every invariant compute() supports has exactly one benchmark row, and every row
    claiming a native compute names one compute() actually accepts."""
    native_rows = [row.tetra[1] for row in spec.INVARIANTS if row.tetra is not None]
    assert len(native_rows) == len(set(native_rows))
    assert sorted(native_rows) == sorted(_supported())
