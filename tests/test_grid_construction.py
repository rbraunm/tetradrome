# SPDX-License-Identifier: Apache-2.0

"""Grid diagrams built from crossing diagrams, and the moves that shrink them.

The knot a grid carries is checked against invariants already validated against KnotInfo: the
grid's graded Euler characteristic is (1 - t)^(n-1) times the Alexander polynomial, and its
HFK-hat, which also fixes chirality, is KnotInfo's. Diagram reductions are checked by the
Jones polynomial, which also sees chirality.
"""
import ast
import random
from collections import defaultdict
from math import comb

import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.diagrams.reduction import reduced, relabellings
from tetradrome.engines.floer import alexander_euler_characteristic, hfk_hat
from tetradrome.engines.floer.grid import GridDiagram
from tetradrome.engines.floer.grid_construction import built, drawing, grid_from_pd
from tetradrome.engines.floer.grid_moves import commutations, destabilized, simplified
from tetradrome.invariants.jones import jones_polynomial
from tetradrome.invariants.seifert import alexander_polynomial, seifert_matrix_from_braid


def _table(most_crossings):
    return [row for row in ki.rows()
            if row.get("pd_notation") and row.get("grid_notation") and row.get("crossing_number")
            and 3 <= int(row["crossing_number"]) <= most_crossings]


def _pd(name):
    return ki.pd_notation(name)


def _normalized(coefficients):
    items = {degree: c for degree, c in coefficients.items() if c}
    low = min(items)
    sequence = [items.get(degree + low, 0) for degree in range(max(items) - low + 1)]
    return tuple(-c for c in sequence) if sequence[0] < 0 else tuple(sequence)


def _expected_euler(name, size):
    delta = dict(enumerate(alexander_polynomial(seifert_matrix_from_braid(ki.braid_word(name)))))
    factor = {i: (-1) ** i * comb(size - 1, i) for i in range(size)}
    product = defaultdict(int)
    for a, ca in factor.items():
        for b, cb in delta.items():
            product[a + b] += ca * cb
    return _normalized(product)


def _carries(grid, name):
    return _normalized(alexander_euler_characteristic(grid)) == _expected_euler(name, grid.n)


# --- reduction ----------------------------------------------------------------------------

def _with_kink(pd, label):
    """``pd`` with a first-Reidemeister kink put on arc ``label``."""
    crossings = [list(crossing) for crossing in pd]
    top = max(max(crossing) for crossing in crossings)
    onward, loop = top + 1, top + 2
    for crossing in crossings:
        if label in crossing:
            crossing[crossing.index(label)] = onward
            break
    crossings.append([label, loop, loop, onward])
    return crossings


def _summed_through_nugatory(first, second):
    """``first`` and ``second`` joined through a nugatory crossing: the first knot's arc 1 runs
    into it and on into the second knot, which comes back through it into the first."""
    offset = max(max(crossing) for crossing in first)
    second = [[label + offset for label in crossing] for crossing in second]
    top = max(max(crossing) for crossing in second)
    resumed = top + 1
    out_of_second = second[0][2]
    first = [list(crossing) for crossing in first]
    for crossing in first:
        if 1 in crossing:
            crossing[crossing.index(1)] = resumed
            break
    renamed = top + 2
    for crossing in second:
        if out_of_second in crossing:
            crossing[crossing.index(out_of_second)] = renamed
            break
    return first + second + [[1, resumed, renamed, out_of_second]]


def test_a_kink_is_removed_and_the_knot_kept():
    for name in ("3_1", "5_2", "8_19"):
        kinked = _with_kink(_with_kink(_pd(name), 2), 5)
        assert len(reduced(kinked)) == len(_pd(name))
        assert jones_polynomial(reduced(kinked)) == jones_polynomial(_pd(name))


def test_a_nugatory_crossing_is_removed_and_the_knot_kept():
    summed = _summed_through_nugatory(_pd("3_1"), _pd("4_1"))
    assert len(summed) == 8
    jones = jones_polynomial(relabellings(summed)[0])
    assert len(reduced(summed)) == 7
    assert jones_polynomial(reduced(summed)) == jones


def test_a_nugatory_crossing_between_chiral_knots_keeps_each_side_s_chirality():
    # The same trefoil twice is the granny knot; had the reduction mirrored one side it would be
    # the square knot, whose Jones polynomial differs.
    granny = _summed_through_nugatory(_pd("3_1"), _pd("3_1"))
    reduced_granny = reduced(granny)
    assert len(reduced_granny) == 6
    assert jones_polynomial(reduced_granny) == jones_polynomial(relabellings(granny)[0])


def test_relabelling_does_not_change_the_reduced_code():
    pd = _pd("7_4")
    for code in relabellings(pd)[:: 5]:
        assert reduced(code) == reduced(pd)


def test_a_diagram_of_the_unknot_with_only_kinks_reduces_to_nothing():
    assert reduced([[1, 2, 2, 1]]) == ()
    assert grid_from_pd([[1, 2, 2, 1]]).n == 2


def test_a_link_is_refused():
    with pytest.raises(ValueError, match="2-component link"):
        reduced([[1, 3, 2, 4], [3, 1, 4, 2]])


# --- construction -------------------------------------------------------------------------

def test_every_horizontal_segment_crosses_exactly_the_column_it_names():
    # The arc presentation is read from the drawing on that promise: each horizontal segment
    # meets at most one vertical segment, and the drawing has no other crossings.
    for row in _table(10):
        drawn = drawing(reduced(ast.literal_eval(row["pd_notation"])))
        rank = {column: x for x, column in enumerate(drawn.columns)}
        for height, (one, other, crossed, _) in enumerate(drawn.horizontals):
            low, high = sorted((rank[one], rank[other]))
            met = [column for column in drawn.columns[low + 1:high]
                   if drawn.spans[column][0] < height < drawn.spans[column][1]]
            assert met == ([] if crossed is None else [crossed]), (row["name"], height)


def test_the_built_grid_has_two_more_rows_than_the_reduced_diagram_has_crossings():
    for row in _table(10):
        diagram = reduced(ast.literal_eval(row["pd_notation"]))
        size = built(diagram).n
        assert size == len(diagram) + 2, row["name"]
        if row["alternating"] == "Y":
            # c + 2 is an alternating knot's smallest grid, which KnotInfo records.
            assert size == len(ast.literal_eval(row["grid_notation"])) // 2, row["name"]


def test_built_grids_carry_the_knot():
    names = [row["name"] for row in _table(5)] + ["6_1", "6_2", "6_3"]
    for name in names:
        grid = grid_from_pd(_pd(name))
        assert grid.n <= 8
        assert _carries(grid, name), name


@pytest.mark.parametrize("name", ["3_1", "5_1", "5_2", "6_2"])
def test_built_grids_give_knotinfo_s_hfk_with_its_chirality(name):
    assert hfk_hat(grid_from_pd(_pd(name))) == ki.hfk_ranks(name)


# --- moves --------------------------------------------------------------------------------

def _markers(grid):
    return tuple(grid.O), tuple(grid.X)


def _grid(markers):
    return GridDiagram(list(markers[0]), list(markers[1]))


def _stabilized(markers, column, row_after):
    """The grid one larger: ``column`` split in two neighbouring columns, its marker in the
    lower-numbered row staying and the other moving right, joined by a new row after
    ``row_after``. The inverse of a destabilization."""
    o_markers, x_markers = markers
    rows = sorted(row for row in range(len(o_markers)) if column in (o_markers[row], x_markers[row]))
    mover = rows[1]
    moved_is_o = o_markers[mover] == column

    def shifted(row, current):
        if current == column:
            return column + 1 if row == mover else column
        return current + 1 if current > column else current

    new_o = [shifted(row, o_markers[row]) for row in range(len(o_markers))]
    new_x = [shifted(row, x_markers[row]) for row in range(len(o_markers))]
    # Column `column` keeps the marker in rows[0]; the new row takes the other kind there and
    # the moved marker's other kind in the column to its right.
    new_o.insert(row_after + 1, column if moved_is_o else column + 1)
    new_x.insert(row_after + 1, column + 1 if moved_is_o else column)
    return tuple(new_o), tuple(new_x)


def test_commutations_keep_the_knot():
    # A smallest grid can be rigid; a twice-stabilized trefoil has room to move.
    markers = _stabilized(_stabilized(_markers(GridDiagram.from_knotinfo("3_1")), 0, 0), 3, 4)
    generator = random.Random(20261119)
    for _ in range(12):
        moves = list(commutations(markers))
        assert moves
        markers = generator.choice(moves)
        assert _carries(_grid(markers), "3_1")


@pytest.mark.parametrize("column, row_after", [(0, 0), (2, 3), (4, 1)])
def test_a_destabilization_removes_one_row_and_keeps_the_knot(column, row_after):
    trefoil = _markers(GridDiagram.from_knotinfo("3_1"))
    larger = _stabilized(trefoil, column, row_after)
    assert _carries(_grid(larger), "3_1")
    smaller = destabilized(larger)
    assert len(smaller[0]) == 5
    assert _carries(_grid(smaller), "3_1")
    assert destabilized(_markers(GridDiagram.from_knotinfo("8_19"))) is None   # already smallest


@pytest.mark.parametrize("name", ["8_19", "8_20", "8_21"])
def test_shrinking_reaches_knotinfo_s_grid_size(name):
    grid = simplified(built(reduced(_pd(name))), 1000)
    assert grid.n == len(ki.grid_notation(name)) // 2
    assert hfk_hat(grid) == ki.hfk_ranks(name)
