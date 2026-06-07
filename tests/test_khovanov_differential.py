"""Tests for the Khovanov differential (engines/khovanov/differential.py).

The decisive check: d^2 = 0 on the assembled complexes. If the merge/split
classification, the circle matching, or the Frobenius maps were wrong, the composite
would not vanish -- so passing verify_d_squared() across a range of knots is strong
evidence the whole differential is right. We also confirm the complexes account for
exactly the generators the grading layer counts.
"""
import pytest

from tetradrome import knots
from tetradrome.engines import khovanov

# Enhanced-state enumeration is exponential, so keep the differential tests small.
KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1"]


@pytest.mark.parametrize("name", KNOTS)
def test_d_squared_is_zero(name):
    pd = knots.from_name(name).pd_code
    complexes = khovanov.khovanov_complexes(pd)
    assert complexes  # at least one quantum grading
    for cx in complexes.values():
        cx.verify_d_squared()  # must not raise


@pytest.mark.parametrize("name", KNOTS)
def test_complexes_account_for_every_generator(name):
    pd = knots.from_name(name).pd_code
    complexes = khovanov.khovanov_complexes(pd)
    total = sum(cx.total_dim() for cx in complexes.values())
    assert total == khovanov.unreduced_size(pd)


def test_per_grading_dimensions_match_chain_dimensions():
    # The (i, j) dimensions inside the complexes must equal the standalone grading count.
    pd = knots.from_name("5_2").pd_code
    expected = khovanov.chain_dimensions(pd)
    got: dict[tuple[int, int], int] = {}
    for j, cx in khovanov.khovanov_complexes(pd).items():
        for i in cx.degrees():
            got[(i, j)] = cx.dim(i)
    assert got == expected


def test_empty_diagram_rejected():
    with pytest.raises(ValueError, match=r"empty diagram"):
        khovanov.khovanov_complexes(())
