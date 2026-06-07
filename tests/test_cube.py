import pytest

from tetradrome import knots
from tetradrome.engines import cube


def test_states_enumerates_all():
    assert list(cube.states(2)) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert len(list(cube.states(5))) == 32


def test_resolve_partitions_labels():
    pd = knots.from_name("3_1").pd_code
    labels = {x for crossing in pd for x in crossing}
    for state in cube.states(len(pd)):
        circles = cube.resolve(pd, state)
        assert set().union(*circles) == labels          # cover every label
        assert sum(len(c) for c in circles) == len(labels)  # ... disjointly


def test_circle_counts_trefoil():
    pd = knots.from_name("3_1").pd_code
    assert cube.circle_count(pd, (0, 0, 0)) == 3  # all A-smoothings
    assert cube.circle_count(pd, (1, 1, 1)) == 2  # all B-smoothings


def test_resolve_rejects_bad_state_length():
    pd = knots.from_name("3_1").pd_code
    with pytest.raises(ValueError):
        cube.resolve(pd, (0, 0))
