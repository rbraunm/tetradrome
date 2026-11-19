# SPDX-License-Identifier: Apache-2.0

"""The knot Floer complex over F2[U, V]/(UV) from the curved bordered engine
(arXiv:1707.00597): the complexes the paper and the oracle give, HFK-hat against ``hfk_hat`` and
KnotInfo, tau against KnotInfo and the grid engine, nu and epsilon against knot_floer_homology
and KnotInfo, and their invariance under the presentation, the orientation, cancellation and
the mirror."""
import ast
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import pytest

from tetradrome.algebra.complex import GradedComplex
from tetradrome.algebra.reduce_f2_packed import pack_columns
from tetradrome.algebra.reduce_reference import f2_rank, homology
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.diagrams.morse import MorseDiagram, morse_diagram
from tetradrome.diagrams.reduction import reduced
from tetradrome.engines.bordered import KnotComplex, hfk_hat, knot_complex
from tetradrome.engines.bordered.algebra import CurvedAlgebra
from tetradrome.engines.bordered.cancellation import cancelled
from tetradrome.engines.bordered.crossing import CrossingBimodule
from tetradrome.engines.bordered.curved_minimum import (CurvedMinimumBimodule,
                                                        curved_minimum_weight_budget)
from tetradrome.engines.bordered.knot import curved_type_d_above_minimum
from tetradrome.engines.bordered.maximum import MaximumBimodule
from tetradrome.engines.bordered.tensor import box
from tetradrome.engines.bordered.type_d import TypeD
from tetradrome.errors import TetradromeError


def _morse(name):
    return morse_diagram(reduced(ki.pd_notation(name)))


def _rows(fewest, most):
    return [row for row in ki.rows()
            if row.get("pd_notation") and row.get("hfk_polynomial_vector")
            and row.get("crossing_number") and fewest <= int(row["crossing_number"]) <= most]


def _four_plat(over):
    """The 4-plat closure of three crossings between the two middle strands: one cup and one
    cap besides the global ones."""
    return MorseDiagram((("min", 0), ("min", 0)) + (("cross", 1, over),) * 3
                        + (("max", 2), ("max", 0)))


def _mirror(morse):
    return MorseDiagram(tuple(piece if piece[0] != "cross" else (piece[0], piece[1], not piece[2])
                              for piece in morse.pieces))


def _kinked(morse, level, position, over, left):
    if left:
        loop = (("min", position), ("cross", position + 1, over), ("max", position))
    else:
        loop = (("min", position + 1), ("cross", position, over), ("max", position + 1))
    return MorseDiagram(morse.pieces[:level] + loop + morse.pieces[level:])


def _twisted(morse, level, position, over):
    pair = (("cross", position, over), ("cross", position, not over))
    return MorseDiagram(morse.pieces[:level] + pair + morse.pieces[level:])


def _mirrored(pd):
    """The PD code of the mirror: the diagram reflected in a line of its plane."""
    return [[a, d, c, b] for a, b, c, d in pd]


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


def _arrows_by_grading(complex_):
    """The arrows named by the gradings of their ends, for a complex with one generator in each
    bigrading."""
    assert len(set(complex_.gradings)) == len(complex_.gradings)
    return {(complex_.gradings[source], complex_.gradings[target], u, v)
            for source, target, u, v in complex_.arrows}


def _oracle(pd, **options):
    import knot_floer_homology

    return knot_floer_homology.pd_to_hfk([list(crossing) for crossing in pd], **options)


def _oracle_complex(pd):
    """knot_floer_homology's UV = 0 complex as a ``KnotComplex``: its generators carry
    (Alexander, Maslov), and the power of U or V on an arrow is the change in Alexander
    grading."""
    found = _oracle(pd, complex=True)
    names = sorted(found["generators"])
    position = {name: index for index, name in enumerate(names)}
    gradings = tuple((found["generators"][name][1], found["generators"][name][0])
                     for name in names)
    arrows = []
    for (source, target), coefficient in found["differentials"].items():
        assert coefficient % 2 == 1
        rise = found["generators"][target][0] - found["generators"][source][0]
        arrows.append((position[source], position[target], max(rise, 0), max(-rise, 0)))
    return KnotComplex(gradings, tuple(arrows))


def _homology_dimensions(complex_, keep, degree, members):
    """{degree: dim H} of the F2 complex on ``members`` with the arrows ``keep`` admits."""
    by_degree = defaultdict(list)
    for generator in members:
        by_degree[degree(generator)].append(generator)
    place = {generator: index for group in by_degree.values()
             for index, generator in enumerate(group)}
    out = defaultdict(list)
    for source, target, u, v in complex_.arrows:
        if source in place and keep(source, u, v):
            assert target in place
            out[source].append(place[target])
    maps = {}
    for value, group in by_degree.items():
        columns = [out[generator] for generator in group]
        if any(columns):
            maps[-value] = columns
    graded = GradedComplex({-value: len(group) for value, group in by_degree.items()}, maps)
    return {-value: dimension for value, dimension in homology(graded).items()}


def _fingerprint(complex_):
    """Homotopy invariants of a complex over F2[U, V]/(UV): for each Alexander grading s, the
    homology of its parts in s of C, C/U and C/V, by degree. Beyond the Alexander gradings of
    HFK-hat they no longer change, so s runs one past them on each side."""
    gradings = complex_.gradings
    alexanders = [alexander for _, alexander in complex_.hat_ranks()]
    found = {}
    everyone = range(len(gradings))
    for s in range(min(alexanders) - 1, max(alexanders) + 2):
        found[("C", s)] = _homology_dimensions(
            complex_,
            lambda g, u, v, s=s: (not u or gradings[g][1] >= s) and (not v or gradings[g][1] <= s),
            lambda g, s=s: gradings[g][0] - 2 * max(0, gradings[g][1] - s), everyone)
        found[("C/U", s)] = _homology_dimensions(
            complex_, lambda g, u, v: not u, lambda g: gradings[g][0],
            [g for g in everyone if gradings[g][1] <= s])
        found[("C/V", s)] = _homology_dimensions(
            complex_, lambda g, u, v: not v, lambda g: gradings[g][0] - 2 * gradings[g][1],
            [g for g in everyone if gradings[g][1] >= s])
    return found


def _invariants(complex_):
    return complex_.tau(), complex_.nu(), complex_.epsilon()


# --- the paper's complexes ----------------------------------------------------------------

def test_the_unknot_is_one_generator_with_no_differential():
    complex_ = knot_complex(morse_diagram(()))
    assert complex_.gradings == ((0, 0),) and complex_.arrows == ()
    assert _invariants(complex_) == (0, 0, 0)


@pytest.mark.parametrize("cancel", [True, False])
def test_the_four_plat_trefoils_are_the_paper_s_complexes(cancel):
    # Right-handed: d b = U a + V c with a, b, c at (M, A) = (0, 1), (-1, 0), (-2, -1).
    right = knot_complex(_four_plat(True), cancel=cancel)
    assert sorted(right.gradings) == [(-2, -1), (-1, 0), (0, 1)]
    assert _arrows_by_grading(right) == {((-1, 0), (0, 1), 1, 0), ((-1, 0), (-2, -1), 0, 1)}
    assert _invariants(right) == (1, 1, 1)
    # Left-handed: d a* = U b*, d c* = V b*.
    left = knot_complex(_four_plat(False), cancel=cancel)
    assert sorted(left.gradings) == [(0, -1), (1, 0), (2, 1)]
    assert _arrows_by_grading(left) == {((0, -1), (1, 0), 1, 0), ((2, 1), (1, 0), 0, 1)}
    assert _invariants(left) == (-1, 0, -1)
    assert right.horizontal == tuple((s, t, u) for s, t, u, v in right.arrows if u)
    assert len(right.horizontal) == len(right.vertical) == 1


def test_t_2_5_is_a_staircase():
    complex_ = knot_complex(_morse("5_1"))
    assert sorted(complex_.gradings) == [(-4, -2), (-3, -1), (-2, 0), (-1, 1), (0, 2)]
    assert _arrows_by_grading(complex_) == {
        ((-1, 1), (0, 2), 1, 0), ((-1, 1), (-2, 0), 0, 1),
        ((-3, -1), (-2, 0), 1, 0), ((-3, -1), (-4, -2), 0, 1)}
    assert _invariants(complex_) == (2, 2, 1)


def test_the_figure_eight_is_a_square_beside_one_generator():
    complex_ = knot_complex(_morse("4_1"))
    assert sorted(complex_.gradings) == [(-1, -1), (0, 0), (0, 0), (0, 0), (1, 1)]
    middle = [g for g, grading in enumerate(complex_.gradings) if grading == (0, 0)]
    ends = [g for g, grading in enumerate(complex_.gradings) if grading != (0, 0)]
    leaving = {g: set() for g in middle}
    entering = {g: set() for g in ends}
    for source, target, u, v in complex_.arrows:
        if source in leaving:
            leaving[source].add(ends.index(target))
        else:
            entering[source].add(middle.index(target))
    # The only change of basis that keeps the gradings mixes the three generators at (0, 0), so
    # the ranks of the arrows from them to the corners and into them from the corners are fixed:
    # one each for a square and a lone generator, two for two zigzags.
    assert f2_rank(pack_columns(leaving.values())) == 1
    assert f2_rank(pack_columns(entering.values())) == 1
    assert _invariants(complex_) == (0, 0, 0)


@pytest.mark.parametrize("first,second,expected", [
    ("3_1", "3_1", (2, 2, 1)), ("3_1", "-3_1", (0, 0, 0)), ("3_1", "4_1", (1, 1, 1)),
    ("-3_1", "5_2", (0, 0, 0))])
def test_connected_sums_add_tau(first, second, expected):
    def code(name):
        return _mirrored(ki.pd_notation(name[1:])) if name[0] == "-" else ki.pd_notation(name)

    summed = reduced(_summed_through_nugatory(code(first), code(second)))
    complex_ = knot_complex(morse_diagram(summed))
    oracle = _oracle(summed)
    assert _invariants(complex_) == expected == (oracle["tau"], oracle["nu"], oracle["epsilon"])
    assert len(complex_.gradings) == oracle["total_rank"]


# --- invariance ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["6_2", "6_3", "7_4", "8_19", "8_20", "9_42"])
def test_the_unreduced_complex_has_the_cancelled_one_s_invariants(name):
    morse = _morse(name)
    full, small = knot_complex(morse, cancel=False), knot_complex(morse)
    assert len(full.gradings) > len(small.gradings)
    assert full.hat_ranks() == small.hat_ranks() == ki.hfk_ranks(name)
    assert _invariants(full) == _invariants(small)
    assert _fingerprint(full) == _fingerprint(small)


def test_the_unreduced_complex_agrees_on_a_kinked_presentation():
    kinked = _kinked(_twisted(_morse("5_2"), 3, 1, True), 5, 2, False, False)
    full, small = knot_complex(kinked, cancel=False), knot_complex(kinked)
    assert _fingerprint(full) == _fingerprint(small) == _fingerprint(knot_complex(_morse("5_2")))


@pytest.mark.parametrize("name,level,position,over,left", [
    ("3_1", 3, 0, True, False), ("4_1", 5, 1, True, True), ("5_2", 4, 2, False, False),
    ("6_2", 6, 2, True, True), ("7_4", 2, 1, False, False)])
def test_a_kink_keeps_the_complex(name, level, position, over, left):
    morse = _morse(name)
    kinked = _kinked(morse, level, position, over, left)
    assert _fingerprint(knot_complex(kinked)) == _fingerprint(knot_complex(morse))


@pytest.mark.parametrize("name,level,position,over", [
    ("3_1", 3, 1, True), ("5_2", 5, 0, True), ("7_6", 6, 3, False)])
def test_a_second_reidemeister_pair_keeps_the_complex(name, level, position, over):
    morse = _morse(name)
    twisted = _twisted(morse, level, position, over)
    assert _fingerprint(knot_complex(twisted)) == _fingerprint(knot_complex(morse))


@pytest.mark.parametrize("name", ["3_1", "5_2", "6_2", "8_19", "9_42", "10_132"])
def test_the_mirror_s_complex_is_the_dual(name):
    # Proposition 10.3: reversing every crossing gives Hom(C, R).
    morse = _morse(name)
    complex_, mirror = knot_complex(morse), knot_complex(_mirror(morse))
    assert _fingerprint(mirror) == _fingerprint(complex_.dual())
    tau, nu, epsilon = _invariants(complex_)
    assert mirror.tau() == -tau and mirror.epsilon() == -epsilon
    assert mirror.nu() == -tau + (epsilon == 1)


@pytest.mark.parametrize("name", ["3_1", "4_1", "5_1", "6_2", "7_2", "8_19"])
def test_reversing_the_orientation_exchanges_u_and_v(name):
    # Proposition 10.1: Delta is unchanged, A is negated, U and V change places. The global
    # minimum's left strand then points down, so its U is the other strand's.
    morse = _morse(name)
    reverse = [tuple(not up for up in level) for level in morse.directions()]
    typed = curved_type_d_above_minimum(morse, reverse)
    assert typed.algebra.upward == frozenset({1})
    backwards, forwards = KnotComplex.from_type_d(typed), knot_complex(morse)
    assert backwards.gradings == tuple(
        (maslov - 2 * alexander, -alexander) for maslov, alexander in forwards.gradings)
    assert backwards.arrows == tuple((s, t, v, u) for s, t, u, v in forwards.arrows)
    assert _invariants(backwards) == _invariants(forwards)


def _above_first_minimum(morse):
    """The curved type D structure just above the first non-global minimum read from the top,
    with the cup moved to strands 0 and 1 by the crossings of (7.5)."""
    upward = morse.directions()
    typed = TypeD(CurvedAlgebra(0, 0, frozenset(), ()))
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
def test_no_output_of_the_curved_minimum_lies_beyond_its_weight_budget(name):
    # The budget prunes paths by the largest total weight an output can carry; one twelve
    # half-units larger must give the same operations.
    typed = _above_first_minimum(_morse(name))
    budget = curved_minimum_weight_budget(typed)
    exact = box(CurvedMinimumBimodule(typed.algebra, budget), typed)
    wider = box(CurvedMinimumBimodule(typed.algebra, budget + 12), typed)
    assert len(exact) == len(wider) > 0
    assert exact.arrows == wider.arrows
    assert any(exact.arrows)


def test_a_complex_that_is_not_one_is_refused():
    with pytest.raises(TetradromeError, match="not homogeneous"):
        KnotComplex(((0, 0), (0, 0)), ((0, 1, 0, 0),))
    with pytest.raises(TetradromeError, match="d\\^2"):
        KnotComplex(((0, 0), (-1, 0), (-2, 0)), ((0, 1, 0, 0), (1, 2, 0, 0)))
    with pytest.raises(TetradromeError, match="UV"):
        KnotComplex(((0, 0), (0, 0)), ((0, 1, 1, 1),))
    two_classes = KnotComplex(((0, 0), (0, 0)), ())
    with pytest.raises(TetradromeError, match="HF-hat"):
        two_classes.tau()


# --- oracles ------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _computed(name):
    return knot_complex(morse_diagram(reduced(ki.pd_notation(name))))


_DEFAULT = [row["name"] for row in _rows(3, 9)]


@pytest.mark.parametrize("name", _DEFAULT)
def test_hfk_hat_from_the_complex_matches_hfk_hat_and_knotinfo_through_nine_crossings(name):
    complex_ = _computed(name)
    assert complex_.hat_ranks() == hfk_hat(_morse(name)) == ki.hfk_ranks(name)
    assert len(complex_.gradings) == sum(ki.hfk_ranks(name).values())


@pytest.mark.parametrize("name", _DEFAULT)
def test_tau_matches_knotinfo_through_nine_crossings(name):
    assert _computed(name).tau() == ki.known_answer(name, "ozsvath_szabo_tau")


@pytest.mark.parametrize("name", _DEFAULT)
def test_tau_nu_and_epsilon_match_knot_floer_homology_through_nine_crossings(name):
    oracle = _oracle(ki.pd_notation(name))
    assert _invariants(_computed(name)) == (oracle["tau"], oracle["nu"], oracle["epsilon"])


def _knotinfo_nu(name):
    """KnotInfo's nu, where its table is consistent with its own tau and epsilon: by (11.2)
    and Hom's theorem nu = tau + 1 exactly when epsilon = -1. For the alternating knots with
    epsilon = -1 the table lists nu = tau, against that rule and against knot_floer_homology;
    those entries are not used."""
    row = ki.lookup(name)
    tau, epsilon = int(row["ozsvath_szabo_tau_invariant"]), int(row["epsilon"])
    listed = ast.literal_eval(row["nu"])[0]
    return listed if listed == tau + (epsilon == -1) else None


@pytest.mark.parametrize("name", _DEFAULT)
def test_epsilon_and_nu_match_knotinfo_through_nine_crossings(name):
    complex_ = _computed(name)
    assert complex_.epsilon() == int(ki.lookup(name)["epsilon"])
    listed = _knotinfo_nu(name)
    if listed is not None:
        assert complex_.nu() == listed


@pytest.mark.parametrize("name", _DEFAULT)
def test_the_complex_has_knot_floer_homology_s_invariants_through_nine_crossings(name):
    # The homology of every Alexander part of C, C/U and C/V is a homotopy invariant of the
    # complex, so the two must share it.
    assert _fingerprint(_computed(name)) == _fingerprint(_oracle_complex(ki.pd_notation(name)))


def _summary(pd):
    """What the heavy tier checks of one knot, computed in a worker."""
    start = time.perf_counter()
    complex_ = knot_complex(morse_diagram(reduced(pd)))
    built = time.perf_counter() - start
    return {"hat": complex_.hat_ranks(), "invariants": _invariants(complex_),
            "fingerprint": _fingerprint(complex_), "generators": len(complex_.gradings),
            "hfk_hat": hfk_hat(morse_diagram(reduced(pd))), "seconds": built}


_HEAVY = [row["name"] for row in _rows(10, 12)]
_HEAVY_TESTS = ("test_the_complex_matches_every_oracle_from_ten_through_twelve_crossings",)


@pytest.fixture(scope="module")
def heavy_summaries(request):
    """The summaries of the 10- to 12-crossing knots whose tests were selected, computed across
    the machine's cores."""
    selected = {item.callspec.params["name"] for item in request.session.items
                if getattr(item, "originalname", None) in _HEAVY_TESTS}
    names = [name for name in _HEAVY if name in selected]
    codes = [ast.literal_eval(ki.lookup(name)["pd_notation"]) for name in names]
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        return dict(zip(names, pool.map(_summary, codes, chunksize=1)))


@pytest.mark.heavy
@pytest.mark.parametrize("name", _HEAVY)
def test_the_complex_matches_every_oracle_from_ten_through_twelve_crossings(heavy_summaries,
                                                                            name):
    summary = heavy_summaries[name]
    pd = ki.pd_notation(name)
    oracle = _oracle(pd)
    assert summary["hat"] == summary["hfk_hat"] == ki.hfk_ranks(name)
    assert summary["generators"] == oracle["total_rank"]
    tau, nu, epsilon = summary["invariants"]
    assert tau == ki.known_answer(name, "ozsvath_szabo_tau")
    assert (tau, nu, epsilon) == (oracle["tau"], oracle["nu"], oracle["epsilon"])
    assert epsilon == int(ki.lookup(name)["epsilon"])
    listed = _knotinfo_nu(name)
    if listed is not None:
        assert nu == listed
    assert summary["fingerprint"] == _fingerprint(_oracle_complex(pd))


@pytest.mark.parametrize("raw,name", [
    (lambda: _with_kink(_with_kink(ki.pd_notation("3_1"), 2), 5), "3_1"),
    (lambda: _with_kink(_mirrored(ki.pd_notation("5_2")), 3), None),
    (lambda: _summed_through_nugatory(ki.pd_notation("3_1"), ki.pd_notation("3_1")), None),
    (lambda: _summed_through_nugatory(ki.pd_notation("3_1"), ki.pd_notation("4_1")), None)],
    ids=["3_1-two-kinks", "mirror-5_2-kink", "3_1-sum-3_1-nugatory", "3_1-sum-4_1-nugatory"])
def test_tau_from_a_raw_code_matches_the_grid_engine(raw, name):
    from tetradrome.engines.floer import tau as grid_tau
    from tetradrome.engines.floer.grid_store import knot_grid

    pd = raw()
    expected = grid_tau(knot_grid(pd).grid)
    if name is not None:
        assert expected == ki.known_answer(name, "ozsvath_szabo_tau")
    assert knot_complex(morse_diagram(reduced(pd))).tau() == expected
