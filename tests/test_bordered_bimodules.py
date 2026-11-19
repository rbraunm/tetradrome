# SPDX-License-Identifier: Apache-2.0

"""The DA bimodules of the bordered Floer engine: the structure relation of section 2.6 of
arXiv:1603.06559 on input sequences up to four long, strict unitality, homogeneity of every
operation, and the operations the paper writes out."""
import random
from itertools import product

import pytest

from tetradrome.engines.bordered.algebra import FIELD, Algebra, pack, state_of, unpack
from tetradrome.engines.bordered.crossing import CrossingBimodule
from tetradrome.engines.bordered.maximum import MaximumBimodule
from tetradrome.engines.bordered.minimum import MinimumBimodule
from tetradrome.engines.bordered.rotation import RotatedBimodule, rotated_algebra
from tetradrome.engines.bordered.tensor import Generator


def _elements(algebra, most, states):
    found = []
    upward = sorted(algebra.upward)
    for left in states:
        for right in states:
            far, packed, _ = algebra.pair(left, right)
            if far:
                continue
            relative = unpack(packed, algebra.strands)
            for exponents in product(range(most + 1), repeat=algebra.strands):
                for mask in range(1 << len(upward)):
                    c_strands = state_of(s for index, s in enumerate(upward) if mask >> index & 1)
                    weights = pack(relative[s] + 2 * exponents[s] + 2 * (c_strands >> s & 1)
                                   for s in range(algebra.strands))
                    element = algebra.element(left, right, weights, c_strands)
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
    """The left side of the DA structure relation; zero for a DA bimodule."""
    upper, lower = bimodule.upper, bimodule.lower
    total = set()
    for element, target in _delta(bimodule, generator, inputs):
        for term in lower.differential(element):
            total ^= {(term, target)}
    for index, element in enumerate(inputs):
        for term in upper.differential(element):
            total ^= _delta(bimodule, generator, inputs[:index] + (term,) + inputs[index + 1:])
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


def _check(bimodule, states, most, count, longest, seed):
    """Check the relation and homogeneity on random sequences; return how many had a nonzero
    operation, so a test can see it exercised something."""
    sampler = random.Random(seed)
    elements = _elements(bimodule.upper, most, states)
    busy = 0
    for generator, inputs in _sequences(bimodule, states, elements, sampler, count, longest):
        assert _relation(bimodule, generator, inputs) == set(), (generator, inputs)
        terms = _delta(bimodule, generator, inputs)
        busy += bool(terms)
        for element, target in terms:
            assert element.left == generator.left and element.right == target.left
            maslov_in = bimodule.maslov(generator) + sum(bimodule.upper.maslov(a) for a in inputs)
            assert maslov_in + len(inputs) - 1 == (bimodule.lower.maslov(element)
                                                   + bimodule.maslov(target))
            alexander_in = (bimodule.alexander(generator)
                            + sum(bimodule.upper.alexander(a) for a in inputs))
            assert alexander_in == bimodule.lower.alexander(element) + bimodule.alexander(target)
    return busy


_LOCAL = [(occupied, frozenset(upward)) for occupied in range(4)
          for upward in ((), (0,), (1,), (0, 1))]


@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("occupied,upward", _LOCAL)
def test_the_local_crossing_bimodule_satisfies_the_structure_relation(positive, occupied, upward):
    upper = Algebra(2, occupied, upward)
    bimodule = CrossingBimodule(upper, 0, positive)
    busy = _check(bimodule, upper.states(), 2, 700, 4, seed=occupied * 8 + len(upward))
    assert busy > (0 if occupied in (0, 3) else 100)


_GLOBAL = [(3, 1, (1,), 1), (3, 2, (0, 2), 0), (4, 2, (1, 2), 1), (4, 2, (0, 3), 1),
           (4, 3, (0, 1, 2), 2), (4, 1, (3,), 2), (5, 2, (1, 3), 2)]


@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("strands,occupied,upward,strand", _GLOBAL)
def test_the_crossing_bimodule_between_other_strands_satisfies_the_structure_relation(
        positive, strands, occupied, upward, strand):
    upper = Algebra(strands, occupied, frozenset(upward))
    bimodule = CrossingBimodule(upper, strand, positive)
    assert _check(bimodule, upper.states(), 1, 500, 4, seed=strands * 31 + strand) > 50


@pytest.mark.parametrize("strand,left_upward,upward", [
    (0, True, ()), (1, False, (0,)), (2, True, (1,)), (1, True, (0, 1)), (0, False, (1,))])
def test_the_maximum_bimodule_satisfies_the_structure_relation(strand, left_upward, upward):
    upper = Algebra(2, 1, frozenset(upward))
    bimodule = MaximumBimodule(upper, strand, left_upward)
    assert _check(bimodule, upper.states(), 2, 600, 3, seed=strand) > 100


def test_the_global_maximum_is_a_type_d_structure_on_one_generator():
    for left_upward in (True, False):
        bimodule = MaximumBimodule(Algebra(0, 0, frozenset()), 0, left_upward)
        (generator,) = bimodule.generators(0)
        assert generator.left == state_of([1])
        assert _relation(bimodule, generator, ()) == set()
        (term,) = _delta(bimodule, generator, ())
        loop_strand = 0 if left_upward else 1
        assert unpack(term[0].weights, 2) == (2, 2) and term[0].c_strands == 1 << loop_strand


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


# (strands, occupied, upward, largest exponent, longest path, operations the paths reach)
_MINIMA = [(4, 2, (1, 2), 3, 6, 54), (4, 2, (1, 3), 3, 6, 54), (4, 2, (0, 3), 2, 5, 390),
           (4, 2, (0, 2), 2, 5, 390), (6, 3, (1, 2, 4), 1, 5, 30), (6, 3, (0, 3, 5), 1, 5, 600)]


@pytest.mark.parametrize("strands,occupied,upward,most,longest,reached", _MINIMA)
def test_the_minimum_bimodule_satisfies_the_structure_relation(strands, occupied, upward, most,
                                                                longest, reached):
    upper = Algebra(strands, occupied, frozenset(upward))
    bimodule = MinimumBimodule(upper, None)
    states = [state for state in upper.states() if not state & 1]
    # Every path through Gamma that leaves the outer strands alone, to the given length.
    inner = [element for element in _elements(upper, most, states)
             if not element.weights >> 2 * FIELD and not element.c_strands >> 2]
    busy = 0
    for generator, inputs in _paths(bimodule, states, inner, longest):
        assert _relation(bimodule, generator, inputs) == set(), (generator, inputs)
        busy += bool(_delta(bimodule, generator, inputs))
    assert busy == reached
    # Random sequences that also carry weight and C's on the outer strands.
    assert _check(bimodule, states, 1, 900, 5, seed=strands + len(upward)) > 20


def test_operations_are_strictly_unital():
    cases = [CrossingBimodule(Algebra(4, 2, frozenset({1, 2})), 1, True),
             CrossingBimodule(Algebra(4, 2, frozenset({0, 2})), 1, False),
             MaximumBimodule(Algebra(2, 1, frozenset({1})), 1, True),
             MinimumBimodule(Algebra(4, 2, frozenset({1, 3})), None)]
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




def _states(*chosen):
    return state_of(chosen)


def test_p_delta_one_is_l_i_from_w_and_r_i_plus_1_from_e():
    bimodule = CrossingBimodule(Algebra(4, 2, frozenset({1, 2})), 1, True)
    lower, right = bimodule.lower, _states(1, 3)
    south = Generator("S", right, right)
    west = Generator("W", _states(2, 3), right)
    east = Generator("E", _states(1, 2), right)
    assert bimodule.generators(right) == [south, west, east]
    assert _delta(bimodule, west, ()) == {(lower.left_move(west.left, 1), south)}
    assert _delta(bimodule, east, ()) == {(lower.right_move(east.left, 2), south)}
    assert _delta(bimodule, south, ()) == set()
    north = Generator("N", _states(2, 3), _states(2, 3))
    assert _delta(bimodule, north, ()) == set()


def test_p_delta_two_of_w_on_u1_is_u2_on_w_plus_l1_l2_on_e():
    bimodule = CrossingBimodule(Algebra(2, 2, frozenset()), 0, True)
    upper, lower, right = bimodule.upper, bimodule.lower, _states(0, 2)
    west = Generator("W", _states(1, 2), right)
    east = Generator("E", _states(0, 1), right)
    assert _delta(bimodule, west, (upper.u(right, 0),)) == {
        (lower.u(west.left, 1), west),
        (lower.element(west.left, east.left, pack((1, 1)), 0), east)}


def test_p_delta_three_of_s_on_u1_then_u2_squared():
    # delta^1_3(S, U1, U2^2) = L2 U1^2 (x) E + R1 U1 U2 (x) W.
    bimodule = CrossingBimodule(Algebra(2, 2, frozenset()), 0, True)
    upper, lower, right = bimodule.upper, bimodule.lower, _states(0, 2)
    south = Generator("S", right, right)
    inputs = (upper.u(right, 0), upper.element(right, right, pack((0, 4)), 0))
    east = Generator("E", _states(0, 1), right)
    west = Generator("W", _states(1, 2), right)
    assert _delta(bimodule, south, inputs) == {
        (lower.element(right, east.left, pack((4, 1)), 0), east),
        (lower.element(right, west.left, pack((3, 2)), 0), west)}


def test_p_delta_two_of_n_on_r_i_plus_2_r_i_plus_1():
    # delta^1_2(N, R_{i+2} R_{i+1}) = R_{i+2} R_{i+1} R_i (x) W + R_{i+2} U_i (x) E, here i = 1.
    bimodule = CrossingBimodule(Algebra(4, 3, frozenset()), 0, True)
    upper, lower = bimodule.upper, bimodule.lower
    start, end = _states(0, 1, 2), _states(0, 2, 3)
    north = Generator("N", start, start)
    west = Generator("W", _states(1, 2, 3), end)
    east = Generator("E", _states(0, 1, 3), end)
    feed = upper.element(start, end, pack((0, 1, 1, 0)), 0)
    assert _delta(bimodule, north, (feed,)) == {
        (lower.element(start, west.left, pack((1, 1, 1, 0)), 0), west),
        (lower.element(start, east.left, pack((2, 0, 1, 0)), 0), east)}


def test_p_passes_a_move_away_from_the_crossing_through_every_generator():
    bimodule = CrossingBimodule(Algebra(4, 2, frozenset({3})), 0, True)
    upper, lower = bimodule.upper, bimodule.lower
    checked = 0
    for right in upper.states():
        move = upper.right_move(right, 3)
        if move is None:
            continue
        for generator in bimodule.generators(right):
            (target,) = [g for g in bimodule.generators(move.right) if g.kind == generator.kind]
            assert _delta(bimodule, generator, (move,)) == {
                (lower.element(generator.left, target.left, move.weights, 0), target)}
            checked += 1
    assert checked == 5


def test_p_delta_three_of_s_on_r_i_and_r_i_plus_2_r_i_plus_1_in_either_grouping():
    # delta^1_3(S, R_i, R_{i+2} R_{i+1}) = delta^1_3(S, R_i R_{i+2}, R_{i+1}) = R_i R_{i+2} (x) E.
    bimodule = CrossingBimodule(Algebra(4, 2, frozenset()), 1, True)
    upper, lower = bimodule.upper, bimodule.lower
    start, end = _states(1, 3), _states(3, 4)
    south = Generator("S", start, start)
    east = Generator("E", _states(2, 4), end)
    expected = {(lower.element(start, east.left, pack((0, 1, 0, 1)), 0), east)}
    r_i = upper.right_move(start, 1)
    rest = upper.element(r_i.right, end, pack((0, 0, 1, 1)), 0)
    assert _delta(bimodule, south, (r_i, rest)) == expected
    both = upper.element(start, _states(2, 4), pack((0, 1, 0, 1)), 0)
    assert _delta(bimodule, south, (both, upper.right_move(both.right, 2))) == expected


def test_n_operations_the_paper_lists():
    bimodule = CrossingBimodule(Algebra(4, 2, frozenset()), 1, False)
    upper, lower, right = bimodule.upper, bimodule.lower, _states(1, 3)
    south = Generator("S", right, right)
    west = Generator("W", _states(2, 3), right)
    east = Generator("E", _states(1, 2), right)
    # delta^1_1(S) = R_i (x) W + L_{i+1} (x) E.
    assert _delta(bimodule, south, ()) == {(lower.right_move(right, 1), west),
                                           (lower.left_move(right, 2), east)}
    # delta^1_2(E, R_i) = R_{i+1} R_i (x) N.
    north = Generator("N", _states(2, 3), _states(2, 3))
    assert _delta(bimodule, east, (upper.right_move(right, 1),)) == {
        (lower.element(east.left, north.left, pack((0, 1, 1, 0)), 0), north)}
    # delta^1_3(W, R_i U_i, R_{i+1}) = R_{i+1} U_{i+1} (x) S.
    single = CrossingBimodule(Algebra(4, 1, frozenset()), 1, False)
    west = Generator("W", _states(2), _states(1))
    feeds = (single.upper.element(_states(1), _states(2), pack((0, 3, 0, 0)), 0),
             single.upper.right_move(_states(2), 2))
    south = Generator("S", _states(3), _states(3))
    assert _delta(single, west, feeds) == {
        (single.lower.element(west.left, south.left, pack((0, 0, 3, 0)), 0), south)}


def test_maximum_operations_of_the_summary_diagrams():
    # c = 2: the cap opens strands 1 and 2 of B(4, 2); the left one points up.
    bimodule = MaximumBimodule(Algebra(2, 1, frozenset({0})), 1, True)
    upper, lower = bimodule.upper, bimodule.lower
    assert lower.upward == frozenset({0, 1})
    x_type, y_type = bimodule.generators(_states(1))
    assert (x_type.kind, x_type.left) == ("X", _states(1, 2))
    assert (y_type.kind, y_type.left) == ("Y", _states(2, 3))
    # delta^1_1: R_{c+1} R_c from X to Y and the loop C_c U_{c+1}.
    assert _delta(bimodule, x_type, ()) == {
        (lower.element(x_type.left, y_type.left, pack((0, 1, 1, 0)), 0), y_type),
        (lower.element(x_type.left, x_type.left, pack((0, 2, 2, 0)), 1 << 1), x_type)}
    # delta^1_2: L_{c-1} (x) L_{c-1} from X to Z, nothing from Y on that input.
    (z_left,) = bimodule.generators(_states(0))
    move = upper.left_move(_states(1), 0)
    assert _delta(bimodule, x_type, (move,)) == {(lower.left_move(x_type.left, 0), z_left)}
    assert _delta(bimodule, y_type, (move,)) == set()
    # R_{c+2} (x) R_c from Y to Z.
    (z_right,) = bimodule.generators(_states(2))
    assert _delta(bimodule, y_type, (upper.right_move(_states(1), 1),)) == {
        (lower.right_move(y_type.left, 3), z_right)}


def test_minimum_operations_on_the_paper_s_preferred_sequences():
    # Strand 1 up: (L2, U1, R2) runs R2 -> C2 -> 1 -> R2 in Gamma and gives the identity.
    bimodule = MinimumBimodule(Algebra(4, 2, frozenset({1, 2})), None)
    upper, lower, start = bimodule.upper, bimodule.lower, _states(2, 3)
    (generator,) = bimodule.generators(start)
    assert generator.left == _states(1)
    l2 = upper.left_move(start, 1)
    feeds = (l2, upper.u(l2.right, 0), upper.right_move(l2.right, 1))
    assert _delta(bimodule, generator, feeds) == {(lower.identity(_states(1)), generator)}
    # A move on a strand beyond the cup passes through, two strands lower.
    move = upper.right_move(start, 3)
    (moved,) = bimodule.generators(move.right)
    assert _delta(bimodule, generator, (move,)) == {(lower.right_move(_states(1), 1), moved)}
    # Strand 0 up: the image (U2, C1) of the sequence (U1, C2), and (U2^2, C1, C1).
    exchanged = MinimumBimodule(Algebra(4, 2, frozenset({0, 3})), None)
    upper = exchanged.upper
    (generator,) = exchanged.generators(start)
    unit = exchanged.lower.identity(_states(1))
    c1 = upper.c(start, 0)
    assert _delta(exchanged, generator, (upper.u(start, 1), c1)) == {(unit, generator)}
    squared = upper.element(start, start, pack((0, 4, 0, 0)), 0)
    assert _delta(exchanged, generator, (squared, c1, c1)) == {(unit, generator)}
    assert _delta(exchanged, generator, (squared, c1)) == set()


def test_p_delta_three_carries_c_s_across_in_either_input():
    # delta^1_3(S, C1 R1, C2 R2) = delta^1_3(S, C1 C2 R1, R2) = R1 C1 C2 (x) E.
    bimodule = CrossingBimodule(Algebra(2, 1, frozenset({0, 1})), 0, True)
    upper, lower = bimodule.upper, bimodule.lower
    south = Generator("S", _states(0), _states(0))
    east = Generator("E", _states(1), _states(2))
    expected = {(lower.element(_states(0), _states(1), pack((3, 2)), 3), east)}
    first = upper.element(_states(0), _states(1), pack((3, 0)), 1)
    second = upper.element(_states(1), _states(2), pack((0, 3)), 2)
    assert _delta(bimodule, south, (first, second)) == expected
    both = upper.element(_states(0), _states(1), pack((3, 2)), 3)
    assert _delta(bimodule, south, (both, upper.right_move(_states(1), 1))) == expected


@pytest.mark.parametrize("strands,occupied,upward,most,longest", [
    (4, 2, (1, 2), 3, 6), (4, 2, (0, 3), 2, 5), (6, 3, (1, 2, 4), 1, 5)])
def test_the_rotated_minimum_satisfies_the_structure_relation(strands, occupied, upward, most,
                                                              longest):
    # The minimum on the two rightmost strands, as mho^1 conjugated by the rotation.
    upper = Algebra(strands, occupied, frozenset(upward))
    bimodule = RotatedBimodule(MinimumBimodule(rotated_algebra(upper), None), upper)
    assert bimodule.lower.upward == frozenset(s for s in upward if s < strands - 2)
    top = 1 << strands
    states = [state for state in upper.states() if not state & top]
    inner = [element for element in _elements(upper, most, states)
             if not element.weights >> FIELD * (strands - 2)
             and not element.c_strands >> strands - 2]
    busy = 0
    for generator, inputs in _paths(bimodule, states, inner, longest):
        assert _relation(bimodule, generator, inputs) == set(), (generator, inputs)
        busy += bool(_delta(bimodule, generator, inputs))
    assert busy > 20
    assert _check(bimodule, states, 1, 600, 5, seed=strands * 7) > 20
