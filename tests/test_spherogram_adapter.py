import pytest

from tetradrome import knots
from tetradrome.errors import UnknownKnot


@pytest.mark.parametrize("name,crossings", [("3_1", 3), ("4_1", 4), ("K11n34", 11)])
def test_from_name_constructs_known_knots(name, crossings):
    k = knots.from_name(name)
    assert k.identity == name
    assert k.source_notation == "name"
    assert k.crossing_number == crossings
    assert all(len(entry) == 4 for entry in k.pd_code)


def test_from_name_unknown_raises():
    with pytest.raises(UnknownKnot):
        knots.from_name("not_a_real_knot_zzz")


def test_from_pd_round_trips_a_normalized_diagram():
    k = knots.from_name("4_1")
    again = knots.from_pd(k.pd_code, identity="4_1")
    assert again.pd_code == k.pd_code
    assert again.source_notation == "pd"
    assert again.identity == "4_1"


def test_normalized_diagram_is_frozen():
    k = knots.from_name("3_1")
    with pytest.raises(Exception):
        k.pd_code = ()  # type: ignore[misc]
