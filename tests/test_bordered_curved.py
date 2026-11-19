# SPDX-License-Identifier: Apache-2.0

"""The curved algebras and bimodules of arXiv:1707.00597 (in the curved form of its section 13
and arXiv:1912.01657): the curvature, the curved type D relation, the curved DA structure
relations of every bimodule on short input sequences, strict unitality, homogeneity, and the
operations the papers write out."""
import random
from itertools import product

import pytest

from tetradrome.engines.bordered.algebra import Algebra, CurvedAlgebra, pack, state_of, unpack
from tetradrome.engines.bordered.crossing import CrossingBimodule
from tetradrome.engines.bordered.curved_minimum import CurvedMinimumBimodule
from tetradrome.engines.bordered.maximum import MaximumBimodule
from tetradrome.engines.bordered.rotation import RotatedBimodule, rotated_algebra
from tetradrome.engines.bordered.tensor import Generator
from tetradrome.engines.bordered.type_d import TypeD
from tetradrome.errors import TetradromeError


def _states(*chosen):
    return state_of(chosen)


def _elements(algebra, most, states):
    """Every nonzero pure element between ``states`` whose U-exponents are at most ``most``."""
    found = []
    for left in states:
        for right in states:
            far, packed, _ = algebra.pair(left, right)
            if far:
                continue
            relative = unpack(packed, algebra.strands)
            for exponents in product(range(most + 1), repeat=algebra.strands):
                weights = pack(relative[s] + 2 * exponents[s] for s in range(algebra.strands))
                element = algebra.element(left, right, weights, 0)
                if element is not None:
                    found.append(element)
    return found


def _delta(bimodule, generator, inputs):
    state = bimodule.start(generator)
    for element in inputs:
        state = bimodule.advance(state, element)
        if state is None:
            return set()
    return set(bimodule.emit(state))


def _relation(bimodule, generator, inputs):
    """The left side of the curved DA structure relation (arXiv:1912.01657,
    (eq:CurvedDAbimoduleRelation1) and (eq:CurvedDAbimoduleRelation2)); zero for a curved DA
    bimodule. The input curvature goes in at every gap of the inputs, the output curvature
    only when there are none."""
    upper, lower = bimodule.upper, bimodule.lower
    total = set()
    for index in range(len(inputs) - 1):
        joined = upper.multiply(inputs[index], inputs[index + 1])
        if joined is not None:
            total ^= _delta(bimodule, generator, inputs[:index] + (joined,) + inputs[index + 2:])
    for split in range(len(inputs) + 1):
        for first, middle in _delta(bimodule, generator, inputs[:split]):
            for second, target in _delta(bimodule, middle, inputs[split:]):
                joined = lower.multiply(first, second)
                if joined is not None:
                    total ^= {(joined, target)}
    gaps = [generator.right] + [element.right for element in inputs]
    for position, state in enumerate(gaps):
        for term in upper.curvature(state):
            total ^= _delta(bimodule, generator, inputs[:position] + (term,) + inputs[position:])
    if not inputs:
        total ^= {(term, generator) for term in lower.curvature(generator.left)}
    return total


def _sequences(bimodule, states, elements, sampler, count, longest):
    """Random generators with input sequences, each input chosen mostly among those that keep
    an operation alive."""
    by_left: dict = {}
    for element in elements:
        by_left.setdefault(element.left, []).append(element)
    for _ in range(count):
        right = sampler.choice(states)
        generators = bimodule.generators(right)
        if not generators:
            continue
        generator = sampler.choice(generators)
        state, current, inputs = bimodule.start(generator), right, []
        for _ in range(sampler.randint(0, longest)):
            candidates = by_left.get(current, [])
            if not candidates:
                break
            live = [a for a in candidates
                    if state is not None and bimodule.advance(state, a) is not None]
            element = sampler.choice(live if live and sampler.random() < 0.85 else candidates)
            inputs.append(element)
            state = bimodule.advance(state, element) if state is not None else None
            current = element.right
            if state is not None and bimodule.emit(state) and sampler.random() < 0.7:
                break
        yield generator, tuple(inputs)


def _homogeneous(bimodule, generator, inputs, terms):
    for element, target in terms:
        assert element.left == generator.left and element.right == target.left
        maslov_in = bimodule.maslov(generator) + sum(bimodule.upper.maslov(a) for a in inputs)
        assert maslov_in + len(inputs) - 1 == bimodule.lower.maslov(element) + bimodule.maslov(
            target)
        alexander_in = (bimodule.alexander(generator)
                        + sum(bimodule.upper.alexander(a) for a in inputs))
        assert alexander_in == bimodule.lower.alexander(element) + bimodule.alexander(target)


def _check(bimodule, states, most, count, longest, seed):
    """Check the relation and homogeneity on random sequences; return how many had a nonzero
    operation, so a test can see it exercised something."""
    sampler = random.Random(seed)
    elements = _elements(bimodule.upper, most, states)
    busy = 0
    for generator, inputs in _sequences(bimodule, states, elements, sampler, count, longest):
        assert _relation(bimodule, generator, inputs) == set(), (generator, inputs)
        terms = _delta(bimodule, generator, inputs)
        _homogeneous(bimodule, generator, inputs, terms)
        busy += bool(terms)
    return busy


def _paths(bimodule, states, elements, longest):
    """Every generator with every input sequence of at most ``longest`` inputs that keeps an
    operation alive after each input."""
    by_left: dict = {}
    for element in elements:
        by_left.setdefault(element.left, []).append(element)

    def walk(generator, state, current, inputs):
        yield generator, inputs
        if len(inputs) == longest:
            return
        for element in by_left.get(current, []):
            advanced = bimodule.advance(state, element)
            if advanced is not None:
                yield from walk(generator, advanced, element.right, inputs + (element,))

    for state in states:
        for generator in bimodule.generators(state):
            yield from walk(generator, bimodule.start(generator), state, ())


def _no_ends(algebra):
    """The idempotent states off positions 0 and m, where the knot's structures live."""
    top = 1 << algebra.strands
    return [state for state in algebra.states() if not state & 1 and not state & top]


# --- the curved algebra -------------------------------------------------------------------

def test_a_matching_must_pair_every_strand_once():
    with pytest.raises(TetradromeError, match="not a matching"):
        CurvedAlgebra(4, 2, frozenset({0, 1}), (2, 3, 0, 0))
    with pytest.raises(TetradromeError, match="not a matching"):
        CurvedAlgebra(4, 2, frozenset({0, 1}), (0, 3, 2, 1))
    with pytest.raises(TetradromeError, match="same way"):
        CurvedAlgebra(4, 2, frozenset({0, 2}), (2, 3, 0, 1))


def test_the_curved_algebra_has_no_c_s():
    algebra = CurvedAlgebra(2, 1, frozenset({0}), (1, 0))
    with pytest.raises(TetradromeError, match="has none"):
        algebra.c(_states(1), 0)


def test_curvature_is_u_i_u_j_over_the_pairs_where_the_relations_leave_it():
    algebra = CurvedAlgebra(4, 2, frozenset({0, 1}), (2, 3, 0, 1))
    # At {1, 3} the generating intervals are strands 1, 2 and strands 3, 4 (the paper's
    # numbering), so U_1 U_3 and U_2 U_4 both survive.
    state = _states(1, 3)
    assert algebra.curvature(state) == {
        algebra.element(state, state, pack((2, 0, 2, 0)), 0),
        algebra.element(state, state, pack((0, 2, 0, 2)), 0)}
    # At {2, 3} neither position beside strand 1 is occupied, so U_1 = 0 by (2.3) and only
    # U_2 U_4 is left; at {0, 1} the same kills U_3 and U_4, and with them both terms.
    state = _states(2, 3)
    assert algebra.curvature(state) == {algebra.element(state, state, pack((0, 2, 0, 2)), 0)}
    assert algebra.curvature(_states(0, 1)) == set()
    # Over the top slice, U_1 U_2 = 0 at {1}: the global maximum is a curved type D structure
    # with no arrows.
    assert CurvedAlgebra(2, 1, frozenset({1}), (1, 0)).curvature(_states(1)) == set()


@pytest.mark.parametrize("partner,upward", [((2, 3, 0, 1), (0, 1)), ((3, 2, 1, 0), (0, 2)),
                                             ((1, 0, 3, 2), (1, 2))])
def test_curvature_is_central_and_of_maslov_minus_two(partner, upward):
    algebra = CurvedAlgebra(4, 2, frozenset(upward), partner)
    elements = _elements(algebra, 1, algebra.states())
    checked = 0
    for element in elements:
        before = {algebra.multiply(term, element) for term in algebra.curvature(element.left)}
        after = {algebra.multiply(element, term) for term in algebra.curvature(element.right)}
        assert before - {None} == after - {None}
        checked += bool(before - {None})
    assert checked > 50
    for state in algebra.states():
        for term in algebra.curvature(state):
            assert (algebra.maslov(term), algebra.alexander(term)) == (-2, 0)


def test_the_type_d_check_demands_delta_squared_be_the_curvature():
    algebra = CurvedAlgebra(4, 2, frozenset({0, 1}), (2, 3, 0, 1))
    state = _states(1, 3)
    typed = TypeD(algebra)
    typed.add_generator(state, 0, 0)
    with pytest.raises(TetradromeError, match="relation fails"):
        typed.check()
    flat = TypeD(algebra)
    flat.add_generator(_states(0, 1), 0, 0)
    flat.check()


def test_the_uncurved_algebra_still_checks_without_curvature():
    typed = TypeD(Algebra(4, 2, frozenset({0, 1})))
    typed.add_generator(_states(1, 3), 0, 0)
    typed.check()


# --- crossings ----------------------------------------------------------------------------

# (strands, occupied, upward, partner, crossing strand): the crossing's strands matched to each
# other or not, at both ends and in the middle.
_CROSSINGS = [(4, 2, (0, 1), (2, 3, 0, 1), 1), (4, 2, (0, 1), (3, 2, 1, 0), 1),
              (4, 2, (0, 1), (2, 3, 0, 1), 0), (4, 2, (0, 2), (1, 0, 3, 2), 0),
              (4, 2, (0, 3), (2, 3, 0, 1), 2), (4, 1, (1, 3), (3, 2, 1, 0), 0),
              (4, 3, (0, 1), (3, 2, 1, 0), 2), (6, 3, (0, 1, 4), (3, 2, 1, 0, 5, 4), 2),
              (6, 3, (0, 1, 2), (5, 4, 3, 2, 1, 0), 2)]


@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("strands,occupied,upward,partner,strand", _CROSSINGS)
def test_the_curved_crossing_satisfies_the_curved_structure_relation(
        positive, strands, occupied, upward, partner, strand):
    upper = CurvedAlgebra(strands, occupied, frozenset(upward), partner)
    bimodule = CrossingBimodule(upper, strand, positive)
    assert _check(bimodule, upper.states(), 1, 500, 4, seed=strands * 31 + strand) > 200


@pytest.mark.parametrize("positive", [True, False])
def test_every_short_crossing_sequence_satisfies_the_curved_relation(positive):
    upper = CurvedAlgebra(4, 2, frozenset({0, 1}), (2, 3, 0, 1))
    bimodule = CrossingBimodule(upper, 1, positive)
    busy = 0
    for generator, inputs in _paths(bimodule, upper.states(),
                                    _elements(upper, 1, upper.states()), 2):
        assert _relation(bimodule, generator, inputs) == set(), (generator, inputs)
        busy += bool(_delta(bimodule, generator, inputs))
    assert busy > 100


def test_p_gamma_zero_of_s_gains_u_beta_r_i_and_u_alpha_l_i_plus_1_when_unmatched():
    # Strands 2, 3 (the paper's 3, 4) cross; they are matched to 1 and 4: alpha = 1, beta = 4.
    upper = CurvedAlgebra(6, 3, frozenset({0, 1, 3}), (5, 2, 1, 4, 3, 0))
    bimodule = CrossingBimodule(upper, 2, True)
    lower, right = bimodule.lower, _states(1, 2, 4)
    assert lower.partner == (5, 3, 4, 1, 2, 0)
    south, west, east = bimodule.generators(right)
    assert (south.kind, west.kind, east.kind) == ("S", "W", "E")
    assert (west.left, east.left) == (_states(1, 3, 4), _states(1, 2, 3))
    assert _delta(bimodule, south, ()) == {
        (lower.element(right, west.left, pack((0, 0, 1, 0, 2, 0)), 0), west),
        (lower.element(right, east.left, pack((0, 2, 0, 1, 0, 0)), 0), east)}
    assert _delta(bimodule, west, ()) == {(lower.left_move(west.left, 2), south)}
    assert _delta(bimodule, east, ()) == {(lower.right_move(east.left, 3), south)}
    # Matched to each other, the strands give S no such term.
    matched = CrossingBimodule(CurvedAlgebra(6, 3, frozenset({0, 1, 2}), (5, 4, 3, 2, 1, 0)),
                               2, True)
    assert _delta(matched, Generator("S", right, right), ()) == set()


def test_n_gamma_zero_reverses_p_s():
    upper = CurvedAlgebra(6, 3, frozenset({0, 1, 3}), (5, 2, 1, 4, 3, 0))
    bimodule = CrossingBimodule(upper, 2, False)
    lower, right = bimodule.lower, _states(1, 2, 4)
    south, west, east = bimodule.generators(right)
    # U_beta L_i from W and U_alpha R_{i+1} from E, beside 1603's R_i and L_{i+1} from S.
    assert _delta(bimodule, west, ()) == {
        (lower.element(west.left, right, pack((0, 0, 1, 0, 2, 0)), 0), south)}
    assert _delta(bimodule, east, ()) == {
        (lower.element(east.left, right, pack((0, 2, 0, 1, 0, 0)), 0), south)}
    assert _delta(bimodule, south, ()) == {(lower.right_move(right, 2), west),
                                           (lower.left_move(right, 3), east)}


# --- maxima -------------------------------------------------------------------------------

@pytest.mark.parametrize("strand,left_upward,upward", [
    (0, True, (1,)), (1, False, (0,)), (2, True, (1,)), (1, True, (0,)), (0, False, (1,))])
def test_the_curved_maximum_satisfies_the_curved_structure_relation(strand, left_upward,
                                                                    upward):
    upper = CurvedAlgebra(2, 1, frozenset(upward), (1, 0))
    bimodule = MaximumBimodule(upper, strand, left_upward)
    assert _check(bimodule, upper.states(), 2, 600, 3, seed=strand) > 100


def test_the_curved_maximum_matches_its_new_strands_and_has_no_loop():
    bimodule = MaximumBimodule(CurvedAlgebra(2, 1, frozenset({0}), (1, 0)), 1, True)
    lower = bimodule.lower
    assert lower.partner == (3, 2, 1, 0) and lower.upward == frozenset({0, 1})
    x_type, y_type = bimodule.generators(_states(1))
    assert _delta(bimodule, x_type, ()) == {
        (lower.element(x_type.left, y_type.left, pack((0, 1, 1, 0)), 0), y_type)}
    assert _delta(bimodule, y_type, ()) == {
        (lower.element(y_type.left, x_type.left, pack((0, 1, 1, 0)), 0), x_type)}


def test_the_global_maximum_is_one_generator_with_no_arrows():
    bimodule = MaximumBimodule(CurvedAlgebra(0, 0, frozenset(), ()), 0, True)
    (generator,) = bimodule.generators(0)
    assert generator.left == _states(1)
    assert _delta(bimodule, generator, ()) == set()
    assert _relation(bimodule, generator, ()) == set()


# --- minima -------------------------------------------------------------------------------

# (strands, occupied, upward, partner, largest exponent, longest path, operations reached)
_MINIMA = [(4, 2, (0, 3), (2, 3, 0, 1), 3, 6, 91), (4, 2, (1, 2), (2, 3, 0, 1), 3, 6, 91),
           (4, 2, (0, 2), (3, 2, 1, 0), 3, 6, 146), (6, 3, (0, 2, 3), (4, 2, 1, 5, 0, 3), 1, 5, 53),
           (6, 3, (0, 2, 5), (3, 5, 4, 0, 2, 1), 1, 5, 53)]


@pytest.mark.parametrize("strands,occupied,upward,partner,most,longest,reached", _MINIMA)
def test_the_curved_minimum_satisfies_the_curved_structure_relation(
        strands, occupied, upward, partner, most, longest, reached):
    upper = CurvedAlgebra(strands, occupied, frozenset(upward), partner)
    bimodule = CurvedMinimumBimodule(upper, None)
    states = [state for state in upper.states() if not state & 1]
    # Every path through the graph (13.2) that leaves the outer strands alone, to the length.
    inner = [element for element in _elements(upper, most, states)
             if not any(unpack(element.weights, strands)[2:])]
    busy = 0
    for generator, inputs in _paths(bimodule, states, inner, longest):
        assert _relation(bimodule, generator, inputs) == set(), (generator, inputs)
        terms = _delta(bimodule, generator, inputs)
        _homogeneous(bimodule, generator, inputs, terms)
        busy += bool(terms)
    assert busy == reached
    # Random sequences that also carry weight on the outer strands.
    assert _check(bimodule, states, 1, 900, 5, seed=strands + len(upward)) > 20


def test_curved_minimum_operations_the_papers_write_out():
    # The cup's strands 0 and 1 (the paper's 1 and 2) are matched to 2 and 3, so alpha = 0 and
    # beta = 1 below, and the output matching pairs them.
    upper = CurvedAlgebra(4, 2, frozenset({0, 3}), (2, 3, 0, 1))
    bimodule = CurvedMinimumBimodule(upper, None)
    lower, start = bimodule.lower, _states(2, 3)
    assert lower.partner == (1, 0)
    (generator,) = bimodule.generators(start)
    assert generator.left == _states(1)
    unit = lower.identity(_states(1))
    u_alpha = lower.u(_states(1), 0)
    l2 = upper.left_move(start, 1)
    # gamma_1(U_2) = U_alpha and gamma_1(U_2^2) = U_alpha^2: delta^1_3(Y R_2, U_2, C_1) and
    # delta^1_4(Y R_2, U_2^2, C_1, C_1) of section 12.
    assert _delta(bimodule, generator, (upper.u(start, 1),)) == {(u_alpha, generator)}
    squared = upper.element(start, start, pack((0, 4, 0, 0)), 0)
    assert _delta(bimodule, generator, (squared,)) == {
        (lower.element(_states(1), _states(1), pack((4, 0)), 0), generator)}
    # gamma_3(L_2, U_1, R_2) = 1 and gamma_5(L_2, U_1, U_2, U_1, R_2) = 1.
    u1 = upper.u(l2.right, 0)
    r2 = upper.right_move(l2.right, 1)
    assert _delta(bimodule, generator, (l2, u1, r2)) == {(unit, generator)}
    u2 = upper.u(l2.right, 1)
    assert _delta(bimodule, generator, (l2, u1, u2, u1, r2)) == {(unit, generator)}
    # A move on a strand beyond the cup passes through, two strands lower.
    move = upper.right_move(start, 3)
    (moved,) = bimodule.generators(move.right)
    assert _delta(bimodule, generator, (move,)) == {(lower.right_move(_states(1), 1), moved)}


def test_curved_minimum_inserts_one_u_alpha_and_one_u_beta_for_u_2_l_2_u_1_squared_r_2():
    # gamma_3(U_2 L_2, U_1^2, R_2) = U_alpha U_beta: one C_1 and one C_2 inserted. The cup's
    # strands are matched to 2 and 5, so alpha = 0 and beta = 3 below.
    upper = CurvedAlgebra(6, 3, frozenset({0, 3, 5}), (2, 5, 0, 4, 3, 1))
    bimodule = CurvedMinimumBimodule(upper, None)
    lower, start = bimodule.lower, _states(2, 3, 5)
    assert lower.partner == (3, 2, 1, 0)
    (generator,) = bimodule.generators(start)
    assert generator.left == _states(1, 3)
    middle = _states(1, 3, 5)
    feeds = (upper.element(start, middle, pack((0, 3, 0, 0, 0, 0)), 0),
             upper.element(middle, middle, pack((4, 0, 0, 0, 0, 0)), 0),
             upper.right_move(middle, 1))
    assert _delta(bimodule, generator, feeds) == {
        (lower.element(generator.left, generator.left, pack((2, 0, 0, 2)), 0), generator)}


def test_the_curved_minimum_refuses_a_cup_on_matched_strands():
    with pytest.raises(TetradromeError, match="unmatched"):
        CurvedMinimumBimodule(CurvedAlgebra(4, 2, frozenset({0, 2}), (1, 0, 3, 2)), None)


@pytest.mark.parametrize("strands,occupied,upward,partner,most,longest,reached", [
    (4, 2, (0, 3), (2, 3, 0, 1), 3, 6, 91), (4, 2, (0, 2), (3, 2, 1, 0), 2, 5, 64),
    (6, 3, (0, 2, 3), (4, 2, 1, 5, 0, 3), 1, 5, 53)])
def test_the_rotated_curved_minimum_satisfies_the_curved_structure_relation(
        strands, occupied, upward, partner, most, longest, reached):
    # The minimum on the two rightmost strands, as mho^1 conjugated by the rotation.
    upper = rotated_algebra(CurvedAlgebra(strands, occupied, frozenset(upward), partner))
    bimodule = RotatedBimodule(CurvedMinimumBimodule(rotated_algebra(upper), None), upper)
    top = 1 << strands
    states = [state for state in upper.states() if not state & top]
    inner = [element for element in _elements(upper, most, states)
             if not any(unpack(element.weights, strands)[:strands - 2])]
    busy = 0
    for generator, inputs in _paths(bimodule, states, inner, longest):
        assert _relation(bimodule, generator, inputs) == set(), (generator, inputs)
        busy += bool(_delta(bimodule, generator, inputs))
    assert busy == reached
    assert _check(bimodule, states, 1, 600, 5, seed=strands * 7) > 20


def test_the_reflected_minimum_reproduces_the_lower_diagram_of_1912_01657():
    # Section (subsec:ExA): matching {1, 4}, {2, 3}, cup on strands 3 and 4, read at V = 0:
    # m_2(x, U_1) = U x, m_4(x, R_3, U_4, L_3) = x, m_4(x, R_3, U_4, L_3 U_1) = U x,
    # m_4(x, R_3, U_1 U_4, L_3) = U x, m_2(x, U_3) = U x. Strand 1 points up at the bottom,
    # so it carries U.
    upper = CurvedAlgebra(4, 2, frozenset({0, 2}), (3, 2, 1, 0))
    bimodule = RotatedBimodule(CurvedMinimumBimodule(rotated_algebra(upper), None), upper)
    lower, start = bimodule.lower, _states(1, 2)
    assert lower.partner == (1, 0)
    (generator,) = bimodule.generators(start)
    out = generator.left
    u = lower.u(out, 0)
    r3 = upper.right_move(start, 2)
    u4 = upper.u(r3.right, 3)
    l3 = upper.left_move(r3.right, 2)
    l3_u1 = upper.element(r3.right, start, pack((2, 0, 1, 0)), 0)
    u1_u4 = upper.element(r3.right, r3.right, pack((2, 0, 0, 2)), 0)
    assert _delta(bimodule, generator, (upper.u(start, 0),)) == {(u, generator)}
    assert _delta(bimodule, generator, (r3, u4, l3)) == {(lower.identity(out), generator)}
    assert _delta(bimodule, generator, (r3, u4, l3_u1)) == {(u, generator)}
    assert _delta(bimodule, generator, (r3, u1_u4, l3)) == {(u, generator)}
    assert _delta(bimodule, generator, (upper.u(start, 2),)) == {(u, generator)}


def test_curved_operations_are_strictly_unital():
    cases = [CrossingBimodule(CurvedAlgebra(4, 2, frozenset({0, 1}), (2, 3, 0, 1)), 1, True),
             CrossingBimodule(CurvedAlgebra(4, 2, frozenset({0, 1}), (2, 3, 0, 1)), 1, False),
             MaximumBimodule(CurvedAlgebra(2, 1, frozenset({1}), (1, 0)), 1, True),
             CurvedMinimumBimodule(CurvedAlgebra(4, 2, frozenset({0, 3}), (2, 3, 0, 1)), None)]
    for bimodule in cases:
        upper = bimodule.upper
        states = [state for state in upper.states() if not state & 1]
        elements = _elements(upper, 1, states)
        for state in states:
            for generator in bimodule.generators(state):
                unit = upper.identity(state)
                assert _delta(bimodule, generator, (unit,)) == {
                    (bimodule.lower.identity(generator.left), generator)}
                for element in elements:
                    if element.left == state:
                        assert _delta(bimodule, generator, (unit, element)) == set()
                        closing = upper.identity(element.right)
                        assert _delta(bimodule, generator, (element, closing)) == set()
