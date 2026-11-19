# SPDX-License-Identifier: Apache-2.0

"""HFK-hat from the bordered engine: the paper's stated results (arXiv:1603.06559 sections 6 and
12), its invariance under moves of the Morse presentation, the unreduced reference, and
agreement with KnotInfo and the grid engine."""
import ast
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.diagrams.morse import MorseDiagram, morse_diagram, pd_from_morse
from tetradrome.diagrams.reduction import reduced, relabellings
from tetradrome.engines.bordered import hfk_hat
from tetradrome.engines.bordered.algebra import Algebra
from tetradrome.engines.bordered.cancellation import cancelled
from tetradrome.engines.bordered.crossing import CrossingBimodule
from tetradrome.engines.bordered.knot import homology_of, type_d_above_minimum
from tetradrome.engines.bordered.maximum import MaximumBimodule
from tetradrome.engines.bordered.minimum import MinimumBimodule, minimum_weight_budget
from tetradrome.engines.bordered.tensor import box
from tetradrome.engines.bordered.type_d import TypeD
from tetradrome.invariants.jones import jones_polynomial
from tetradrome.invariants.seifert import alexander_polynomial, seifert_matrix_from_braid, signature

TREFOIL_RIGHT = {(0, 1): 1, (-1, 0): 1, (-2, -1): 1}


def _morse(name):
    return morse_diagram(reduced(ki.pd_notation(name)))


def _rows(fewest, most):
    return [row for row in ki.rows()
            if row.get("pd_notation") and row.get("hfk_polynomial_vector")
            and row.get("crossing_number") and fewest <= int(row["crossing_number"]) <= most]


def _four_plat(over):
    """The 4-plat closure of three crossings between the two middle strands (section 6.2 of the
    implementation notes): one cup and one cap besides the global ones."""
    return MorseDiagram((("min", 0), ("min", 0)) + (("cross", 1, over),) * 3
                        + (("max", 2), ("max", 0)))


def _mirror(morse):
    return MorseDiagram(tuple(piece if piece[0] != "cross" else (piece[0], piece[1], not piece[2])
                              for piece in morse.pieces))


def _kinked(morse, level, position, over, left):
    """``morse`` with a first-Reidemeister loop on the strand at ``position`` of ``level``,
    curling out to the left or to the right."""
    if left:
        loop = (("min", position), ("cross", position + 1, over), ("max", position))
    else:
        loop = (("min", position + 1), ("cross", position, over), ("max", position + 1))
    return MorseDiagram(morse.pieces[:level] + loop + morse.pieces[level:])


def _twisted(morse, level, position, over):
    """``morse`` with two strands of ``level`` crossed and uncrossed (a second Reidemeister
    move)."""
    pair = (("cross", position, over), ("cross", position, not over))
    return MorseDiagram(morse.pieces[:level] + pair + morse.pieces[level:])


def _draws(morse, name):
    return jones_polynomial(pd_from_morse(morse)) == jones_polynomial(
        relabellings(ki.pd_notation(name))[0])


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
    """``first`` and ``second`` joined through a nugatory crossing."""
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


def _tensor(first, second):
    product = defaultdict(int)
    for (m1, a1), r1 in first.items():
        for (m2, a2), r2 in second.items():
            product[(m1 + m2, a1 + a2)] += r1 * r2
    return dict(product)


# --- the paper's stated results -----------------------------------------------------------

def test_the_unknot_has_rank_one_at_the_origin():
    assert hfk_hat(morse_diagram(())) == {(0, 0): 1}


def test_the_four_plat_trefoil_has_its_three_kauffman_states_and_no_differential():
    right, left = _four_plat(True), _four_plat(False)
    # Both middle strands point up at the crossings, so P^2 is the positive crossing.
    assert right.directions()[2] == (False, True, True, False)
    assert _draws(right, "3_1")
    typed = type_d_above_minimum(right, right.directions(), cancel=False)
    assert sorted(zip(typed.maslov, typed.alexander)) == [(-2, -2), (-1, 0), (0, 2)]
    assert hfk_hat(right, cancel=False) == TREFOIL_RIGHT
    assert hfk_hat(left) == {(0, -1): 1, (1, 0): 1, (2, 1): 1}


@pytest.mark.parametrize("name", ["3_1", "4_1", "5_2", "6_3", "7_4", "8_12", "9_23"])
def test_alternating_knots_follow_the_signature_and_alexander_polynomial(name):
    # Proposition 12.1: rank |a_s| at Maslov s + sigma / 2, nothing else.
    seifert = seifert_matrix_from_braid(ki.braid_word(name))
    coefficients, sigma = alexander_polynomial(seifert), signature(seifert)
    top = (len(coefficients) - 1) // 2
    expected = {(s + sigma // 2, s): abs(a)
                for s, a in zip(range(-top, top + 1), coefficients) if a}
    assert ki.lookup(name)["alternating"] == "Y"
    assert hfk_hat(_morse(name)) == expected


@pytest.mark.parametrize("name", ["3_1", "5_2", "6_2", "8_19", "9_42"])
def test_the_mirror_reflects_both_gradings(name):
    # Proposition 12.2.
    morse = _morse(name)
    assert hfk_hat(_mirror(morse)) == {(-m, -a): r for (m, a), r in hfk_hat(morse).items()}


@pytest.mark.parametrize("first,second", [("3_1", "4_1"), ("3_1", "3_1"), ("5_2", "3_1")])
def test_a_connected_sum_is_the_tensor_product(first, second):
    # Proposition 12.3, on the sum drawn through a nugatory crossing and reduced.
    summed = reduced(_summed_through_nugatory(ki.pd_notation(first), ki.pd_notation(second)))
    assert hfk_hat(morse_diagram(summed)) == _tensor(ki.hfk_ranks(first), ki.hfk_ranks(second))


@pytest.mark.parametrize("name", ["3_1", "4_1", "6_1", "8_20", "9_46"])
def test_the_euler_characteristic_is_the_alexander_polynomial(name):
    # Corollary 11.11.
    euler = defaultdict(int)
    for (maslov, alexander), rank in hfk_hat(_morse(name)).items():
        euler[alexander] += (-1) ** maslov * rank
    coefficients = alexander_polynomial(seifert_matrix_from_braid(ki.braid_word(name)))
    top = (len(coefficients) - 1) // 2
    expected = dict(zip(range(-top, top + 1), coefficients))
    sign = 1 if euler[top] == expected[top] else -1
    assert {s: c for s, c in euler.items() if c} == {s: sign * c for s, c in expected.items() if c}


# --- invariance, orientation, and the unreduced reference ---------------------------------

@pytest.mark.parametrize("name,level,position,over,left", [
    ("3_1", 3, 0, True, False), ("3_1", 3, 3, False, True), ("4_1", 5, 1, True, True),
    ("5_2", 4, 2, False, False), ("6_2", 6, 2, True, True), ("7_4", 2, 1, False, False)])
def test_a_kink_in_the_presentation_keeps_the_answer(name, level, position, over, left):
    morse = _morse(name)
    kinked = _kinked(morse, level, position, over, left)
    assert kinked.widths()[level] == morse.widths()[level]
    assert _draws(kinked, name)
    assert hfk_hat(kinked) == ki.hfk_ranks(name)


@pytest.mark.parametrize("name,level,position,over", [
    ("3_1", 3, 1, True), ("4_1", 4, 2, False), ("5_2", 5, 0, True), ("7_6", 6, 3, False)])
def test_a_second_reidemeister_pair_keeps_the_answer(name, level, position, over):
    twisted = _twisted(_morse(name), level, position, over)
    assert _draws(twisted, name)
    assert hfk_hat(twisted) == ki.hfk_ranks(name)


@pytest.mark.parametrize("name", ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_2"])
def test_reversing_the_orientation_gives_the_same_answer(name):
    # The other orientation starts down the left strand of the global minimum (the paper's LR
    # case), so the terminal module is over B(2, 1, {2}).
    morse = _morse(name)
    reverse = [tuple(not up for up in level) for level in morse.directions()]
    typed = type_d_above_minimum(morse, reverse)
    assert typed.algebra.upward == frozenset({1})
    assert homology_of(typed) == ki.hfk_ranks(name)


@pytest.mark.parametrize("name", ["3_1", "4_1", "5_2", "6_1", "6_3", "7_5"])
def test_the_unreduced_computation_agrees_with_the_cancelled_one(name):
    morse = _morse(name)
    assert hfk_hat(morse, cancel=False) == hfk_hat(morse) == ki.hfk_ranks(name)


def test_the_unreduced_computation_agrees_on_a_kinked_presentation():
    kinked = _kinked(_twisted(_morse("4_1"), 3, 1, True), 5, 2, False, False)
    assert hfk_hat(kinked, cancel=False) == hfk_hat(kinked) == ki.hfk_ranks("4_1")


def _above_first_minimum(morse):
    """The type D structure just above the first non-global minimum read from the top, with the
    cup moved to strands 0 and 1 by the crossings of (9.4)."""
    upward = morse.directions()
    typed = TypeD(Algebra(0, 0, frozenset()))
    typed.add_generator(0, 0, 0)
    for level in range(len(morse.pieces) - 1, 0, -1):
        kind, at = morse.pieces[level][0], morse.pieces[level][1]
        if kind == "min":
            for crossing in range(at - 1, -1, -1):
                for strand in (crossing, crossing + 1):
                    typed = cancelled(box(CrossingBimodule(typed.algebra, strand, True), typed))
            return typed
        if kind == "max":
            bimodule = MaximumBimodule(typed.algebra, at, upward[level][at])
        else:
            bimodule = CrossingBimodule(typed.algebra, at, morse.pieces[level][2])
        typed = cancelled(box(bimodule, typed))
    raise AssertionError("no minimum above the global one")


@pytest.mark.parametrize("name", ["6_2", "8_14", "9_44", "10_132"])
def test_no_output_of_the_minimum_lies_beyond_its_weight_budget(name):
    # The budget prunes paths by the largest total weight an output can carry; one twelve
    # half-units larger must give the same operations.
    typed = _above_first_minimum(_morse(name))
    budget = minimum_weight_budget(typed)
    exact = box(MinimumBimodule(typed.algebra, budget), typed)
    wider = box(MinimumBimodule(typed.algebra, budget + 12), typed)
    assert len(exact) == len(wider) > 0
    assert exact.arrows == wider.arrows
    assert any(exact.arrows)


# --- oracles ------------------------------------------------------------------------------

@pytest.mark.parametrize("row", _rows(3, 9), ids=lambda row: row["name"])
def test_hfk_matches_knotinfo_through_nine_crossings(row):
    pd = ast.literal_eval(row["pd_notation"])
    assert hfk_hat(morse_diagram(reduced(pd))) == ki.hfk_ranks(row["name"])


_HEAVY = _rows(10, 12)


@pytest.fixture(scope="module")
def heavy_hfk(request):
    """The bordered HFK-hat of the 10- to 12-crossing knots whose tests were selected, computed
    across the machine's cores."""
    selected = {item.callspec.params["name"] for item in request.session.items
                if getattr(item, "originalname", None) == _HEAVY_TEST}
    rows = [row for row in _HEAVY if row["name"] in selected]
    morses = [morse_diagram(reduced(ast.literal_eval(row["pd_notation"]))) for row in rows]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        return dict(zip((row["name"] for row in rows), pool.map(hfk_hat, morses, chunksize=1)))


_HEAVY_TEST = "test_hfk_matches_knotinfo_from_ten_through_twelve_crossings"


@pytest.mark.heavy
@pytest.mark.parametrize("name", [row["name"] for row in _HEAVY])
def test_hfk_matches_knotinfo_from_ten_through_twelve_crossings(heavy_hfk, name):
    assert heavy_hfk[name] == ki.hfk_ranks(name)


@pytest.mark.parametrize("raw,name", [
    (lambda: _with_kink(_with_kink(ki.pd_notation("3_1"), 2), 5), "3_1"),
    (lambda: _with_kink(ki.pd_notation("5_2"), 3), "5_2"),
    (lambda: _summed_through_nugatory(ki.pd_notation("3_1"), ki.pd_notation("4_1")), None)],
    ids=["3_1-two-kinks", "5_2-kink", "3_1-sum-4_1-nugatory"])
def test_hfk_from_a_raw_code_matches_the_grid_engine(raw, name):
    from tetradrome.engines.floer import hfk_hat as grid_hfk_hat
    from tetradrome.engines.floer.grid_store import knot_grid

    pd = raw()
    expected = grid_hfk_hat(knot_grid(pd).grid)
    if name is not None:
        assert expected == ki.hfk_ranks(name)
    assert hfk_hat(morse_diagram(reduced(pd))) == expected
