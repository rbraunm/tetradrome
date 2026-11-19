# SPDX-License-Identifier: Apache-2.0

"""The algebra B(m, k, S) of the bordered Floer engine: its relations, the small examples the
paper works out, the associative and Leibniz laws, and the gradings (arXiv:1603.06559, section
3)."""
import random
from itertools import product

from tetradrome.engines.bordered.algebra import Algebra, Element, pack, positions, state_of, unpack


def _elements(algebra, most):
    """Every nonzero pure element whose U-exponents are at most ``most``."""
    found = []
    upward = sorted(algebra.upward)
    for left in algebra.states():
        for right in algebra.states():
            far, packed, _ = algebra.pair(left, right)
            relative = unpack(packed, algebra.strands)
            if far:
                continue
            for exponents in product(range(most + 1), repeat=algebra.strands):
                for size in range(len(upward) + 1):
                    for chosen in _subsets(upward, size):
                        c_strands = state_of(chosen)
                        weights = pack(relative[s] + 2 * exponents[s] + 2 * (c_strands >> s & 1)
                                       for s in range(algebra.strands))
                        element = algebra.element(left, right, weights, c_strands)
                        if element is not None:
                            found.append(element)
    return found


def _subsets(items, size):
    if size == 0:
        yield ()
        return
    for index, item in enumerate(items):
        for rest in _subsets(items[index + 1:], size - 1):
            yield (item,) + rest


def _sum_product(algebra, first: set, second: set) -> set:
    total = set()
    for a in first:
        for b in second:
            term = algebra.multiply(a, b)
            if term is not None:
                total ^= {term}
    return total


def test_figure_7_r1_then_l1_is_u1_at_0_2_3():
    algebra = Algebra(4, 3, frozenset())
    start = state_of([0, 2, 3])
    r1 = algebra.right_move(start, 0)
    assert positions(r1.right) == [1, 2, 3]
    l1 = algebra.left_move(r1.right, 0)
    assert algebra.multiply(r1, l1) == algebra.u(start, 0)


def test_moving_one_coordinate_twice_is_zero_and_two_coordinates_is_not():
    single = Algebra(4, 1, frozenset())
    # R_i R_{i+1} = 0: the coordinate at 1 crosses strand 1, then strand 2.
    first = single.right_move(state_of([1]), 1)
    assert single.multiply(first, single.right_move(first.right, 2)) is None
    # L_{i+1} L_i = 0 likewise.
    first = single.left_move(state_of([3]), 2)
    assert single.multiply(first, single.left_move(first.right, 1)) is None
    algebra = Algebra(4, 2, frozenset())
    # R_{i+1} R_i moves two different coordinates and survives.
    start = state_of([1, 2])
    first = algebra.right_move(start, 2)
    second = algebra.right_move(first.right, 1)
    both = algebra.multiply(first, second)
    assert both is not None and positions(both.right) == [2, 3]
    assert unpack(both.weights, 4) == (0, 1, 1, 0)


def test_b_2_1_matches_the_paper_s_worked_relations():
    algebra = Algebra(2, 1, frozenset())
    one, zero = state_of([1]), state_of([0])
    # I_{1} B I_{1} = F[U1, U2] / (U1 U2): U1 and U2 survive, their product does not.
    assert algebra.u(one, 0) is not None and algebra.u(one, 1) is not None
    assert algebra.multiply(algebra.u(one, 0), algebra.u(one, 1)) is None
    # I_{0} B I_{0} = F[U1, U2] / (U2).
    assert algebra.u(zero, 0) is not None and algebra.u(zero, 1) is None
    # R1 L1 = U1 and L1 R1 = U1 at the other end.
    r1 = algebra.right_move(zero, 0)
    l1 = algebra.left_move(one, 0)
    assert algebra.multiply(r1, l1) == algebra.u(zero, 0)
    assert algebra.multiply(l1, r1) == algebra.u(one, 0)
    # The far states {0} and {2} have nothing between them.
    assert algebra.pair(zero, state_of([2]))[0]


def test_b_m_0_is_the_field_and_b_m_m_plus_1_is_a_polynomial_ring():
    empty = Algebra(3, 0, frozenset())
    assert empty.states() == [0]
    assert all(empty.u(0, strand) is None for strand in range(3))
    full = Algebra(3, 4, frozenset())
    (state,) = full.states()
    power = full.identity(state)
    for strand in (0, 1, 2, 0, 2, 1):
        power = full.multiply(power, full.u(state, strand))
    assert power is not None and full.exponents(power) == (2, 2, 2)


def test_u_vanishes_at_an_idempotent_with_neither_neighbouring_position():
    algebra = Algebra(4, 2, frozenset())
    state = state_of([0, 4])
    assert [algebra.u(state, strand) is None for strand in range(4)] == [False, True, True, False]


def test_the_generating_interval_of_two_states_kills_its_u_product():
    # x = y = {1, 2} in B(3, 2): positions 0 and 3 are free, 1 and 2 common, so U1 U2 U3 = 0 while
    # every proper sub-product survives.
    algebra = Algebra(3, 2, frozenset())
    state = state_of([1, 2])
    assert algebra.element(state, state, pack((2, 2, 0)), 0) is not None
    assert algebra.element(state, state, pack((2, 2, 2)), 0) is None


def test_multiplication_is_associative():
    algebra = Algebra(3, 2, frozenset({0, 2}))
    elements = _elements(algebra, 1)
    by_left: dict = {}
    for element in elements:
        by_left.setdefault(element.left, []).append(element)
    sampler = random.Random(3)
    checked = 0
    for _ in range(4000):
        a = sampler.choice(elements)
        b = sampler.choice(by_left[a.right])
        c = sampler.choice(by_left[b.right])
        ab, bc = algebra.multiply(a, b), algebra.multiply(b, c)
        left = algebra.multiply(ab, c) if ab is not None else None
        right = algebra.multiply(a, bc) if bc is not None else None
        assert left == right
        checked += left is not None
    assert checked > 100


def test_the_differential_squares_to_zero_and_obeys_leibniz():
    algebra = Algebra(3, 1, frozenset({1, 2}))
    elements = _elements(algebra, 1)
    by_left: dict = {}
    for element in elements:
        by_left.setdefault(element.left, []).append(element)
    for element in elements:
        twice = set()
        for term in algebra.differential(element):
            twice ^= algebra.differential(term)
        assert twice == set()
    sampler = random.Random(5)
    nontrivial = 0
    for _ in range(3000):
        a = sampler.choice(elements)
        b = sampler.choice(by_left[a.right])
        ab = algebra.multiply(a, b)
        lhs = algebra.differential(ab) if ab is not None else set()
        rhs = _sum_product(algebra, algebra.differential(a), {b}) ^ _sum_product(
            algebra, {a}, algebra.differential(b))
        assert lhs == rhs
        nontrivial += bool(lhs)
    assert nontrivial > 50


def test_gradings_of_the_generators_on_upward_and_downward_strands():
    algebra = Algebra(2, 1, frozenset({0}))
    zero, one = state_of([0]), state_of([1])
    up_move, down_move = algebra.right_move(zero, 0), algebra.right_move(one, 1)
    assert (algebra.maslov(up_move), algebra.alexander(up_move)) == (-1, -1)
    assert (algebra.maslov(down_move), algebra.alexander(down_move)) == (0, 1)
    assert (algebra.maslov(algebra.u(one, 0)), algebra.alexander(algebra.u(one, 0))) == (-2, -2)
    assert (algebra.maslov(algebra.u(one, 1)), algebra.alexander(algebra.u(one, 1))) == (0, 2)
    assert (algebra.maslov(algebra.c(one, 0)), algebra.alexander(algebra.c(one, 0))) == (-1, -2)


def test_gradings_add_under_products_and_the_differential_lowers_maslov_by_one():
    algebra = Algebra(4, 2, frozenset({1, 2}))
    elements = _elements(algebra, 1)
    by_left: dict = {}
    for element in elements:
        by_left.setdefault(element.left, []).append(element)
    sampler = random.Random(7)
    for _ in range(3000):
        a = sampler.choice(elements)
        b = sampler.choice(by_left[a.right])
        ab = algebra.multiply(a, b)
        if ab is not None:
            assert algebra.maslov(ab) == algebra.maslov(a) + algebra.maslov(b)
            assert algebra.alexander(ab) == algebra.alexander(a) + algebra.alexander(b)
        for term in algebra.differential(a):
            assert algebra.maslov(term) == algebra.maslov(a) - 1
            assert algebra.alexander(term) == algebra.alexander(a)


def test_a_nonzero_element_is_fixed_by_its_left_idempotent_and_weight():
    # Proposition 3.9: no two distinct pure elements share a left idempotent, weight and C's.
    algebra = Algebra(4, 2, frozenset({0, 3}))
    seen = {}
    for element in _elements(algebra, 1):
        key = (element.left, element.weights, element.c_strands)
        assert seen.setdefault(key, element) == element


def test_elements_are_named_by_their_data():
    algebra = Algebra(2, 1, frozenset({1}))
    element = algebra.c(state_of([1]), 1)
    assert element == Element(state_of([1]), state_of([1]), pack((0, 2)), 2)
