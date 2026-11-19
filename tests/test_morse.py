# SPDX-License-Identifier: Apache-2.0

"""Morse presentations built from the leveling draw the knot they came from: the PD code read back
from the pieces has the knot's Jones polynomial, which tells most knots from their mirrors."""
import ast

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.diagrams.morse import MorseDiagram, morse_diagram, pd_from_morse
from tetradrome.diagrams.reduction import reduced
from tetradrome.invariants.jones import jones_polynomial


def _rows(most):
    return [row for row in ki.rows()
            if row.get("pd_notation") and row.get("crossing_number")
            and 3 <= int(row["crossing_number"]) <= most]


def test_the_pieces_draw_the_knot_with_its_chirality():
    for row in _rows(9):
        pd = ast.literal_eval(row["pd_notation"])
        morse = morse_diagram(reduced(pd))
        assert jones_polynomial(pd_from_morse(morse)) == jones_polynomial(pd), row["name"]


def test_a_diagram_of_c_crossings_has_c_crossing_pieces_and_closes_up():
    for row in _rows(10):
        diagram = reduced(ast.literal_eval(row["pd_notation"]))
        morse = morse_diagram(diagram)
        kinds = [piece[0] for piece in morse.pieces]
        assert kinds.count("cross") == len(diagram)
        assert kinds.count("min") == kinds.count("max") >= 2
        widths = morse.widths()
        assert widths[0] == widths[-1] == 0


def test_directions_follow_the_knot_through_every_piece():
    for row in _rows(8):
        morse = morse_diagram(reduced(ast.literal_eval(row["pd_notation"])))
        directions = morse.directions()
        for level, piece in enumerate(morse.pieces):
            below, above = directions[level], directions[level + 1]
            kind, at = piece[0], piece[1]
            if kind == "min":
                # One leg of a cup goes down into it and the other comes up out of it.
                assert above[at] != above[at + 1]
                assert above[:at] + above[at + 2:] == below
            elif kind == "max":
                assert below[at] != below[at + 1]
                assert below[:at] + below[at + 2:] == above
            else:
                assert (above[at], above[at + 1]) == (below[at + 1], below[at])
                assert above[:at] + above[at + 2:] == below[:at] + below[at + 2:]


def test_the_unknot_is_one_minimum_and_one_maximum():
    morse = morse_diagram(())
    assert morse.pieces == (("min", 0), ("max", 0))
    assert morse.directions() == [(), (True, False), ()]
    assert isinstance(morse, MorseDiagram)
