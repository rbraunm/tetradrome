# SPDX-License-Identifier: Apache-2.0

"""The grid store: a knot's grid is built once, kept under its reduced diagram, and built again
only when the code that builds grids changes. And the braid closure the Floer path takes a
braid-presented knot through."""
import ast
import json

import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.diagrams.braid_closure import pd_from_braid
from tetradrome.diagrams.reduction import reduced, relabellings
from tetradrome.engines.floer import grid_store
from tetradrome.invariants.jones import jones_polynomial


def _rows(fewest, most):
    return [row for row in ki.rows()
            if row.get("pd_notation") and row.get("grid_notation") and row.get("crossing_number")
            and fewest <= int(row["crossing_number"]) <= most]


def test_a_grid_is_built_once_then_read_back(tmp_path):
    pd = ki.pd_notation("8_20")
    first = grid_store.knot_grid(pd, tmp_path)
    second = grid_store.knot_grid(pd, tmp_path)
    assert first.source == "built"
    assert second.source == "stored"
    assert (second.o_markers, second.x_markers, second.built_at) == (
        first.o_markers, first.x_markers, first.built_at)
    assert second.stored_in == first.stored_in


def test_another_labelling_of_the_same_diagram_finds_the_same_grid(tmp_path):
    pd = ki.pd_notation("7_4")
    first = grid_store.knot_grid(pd, tmp_path)
    relabelled = grid_store.knot_grid(relabellings(pd)[-1], tmp_path)
    assert relabelled.source == "stored"
    assert relabelled.stored_in == first.stored_in


def test_a_grid_from_other_code_is_built_again(tmp_path):
    pd = ki.pd_notation("6_2")
    first = grid_store.knot_grid(pd, tmp_path)
    path = first.stored_in
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    stored["generator"] = "0" * 64
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)
    rebuilt = grid_store.knot_grid(pd, tmp_path)
    assert rebuilt.source == "built"
    assert rebuilt.generator == grid_store.generator_stamp()


def test_an_unreadable_stored_grid_raises_naming_its_file(tmp_path):
    pd = ki.pd_notation("6_2")
    path = grid_store.knot_grid(pd, tmp_path).stored_in
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    with pytest.raises(RuntimeError, match="unreadable stored grid"):
        grid_store.knot_grid(pd, tmp_path)


def test_alternation_is_read_from_the_diagram_as_knotinfo_records_it():
    for row in _rows(3, 10):
        diagram = reduced(ast.literal_eval(row["pd_notation"]))
        assert grid_store._alternating(diagram) == (row["alternating"] == "Y"), row["name"]


def _reaches_knotinfo(fewest, most):
    for row in _rows(fewest, most):
        if row["alternating"] == "Y":
            continue
        grid = grid_store.smallest_grid(reduced(ast.literal_eval(row["pd_notation"])))
        assert grid.n == len(ast.literal_eval(row["grid_notation"])) // 2, row["name"]


def test_the_kept_grid_is_as_small_as_knotinfo_s_through_nine_crossings():
    _reaches_knotinfo(3, 9)


@pytest.mark.heavy
def test_the_kept_grid_is_as_small_as_knotinfo_s_for_ten_and_eleven_crossings():
    _reaches_knotinfo(10, 11)


def test_a_braid_closes_to_the_knot_with_its_chirality():
    # The Jones polynomial tells most knots from their mirrors.
    for row in _rows(3, 9):
        word = ki.braid_word(row["name"])
        assert jones_polynomial(pd_from_braid(word)) == jones_polynomial(
            ast.literal_eval(row["pd_notation"])), row["name"]


def test_a_braid_whose_closure_is_a_link_is_refused():
    with pytest.raises(ValueError, match="2-component link"):
        pd_from_braid([1, 1])
    with pytest.raises(ValueError, match="2-component link"):
        pd_from_braid([2, 2, 2])
