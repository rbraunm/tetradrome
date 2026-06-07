import pytest

from tetradrome import knots
from tetradrome.diagrams import seifert_structure

# Alternating knots, where Seifert's algorithm gives a minimal-genus surface, so the
# Seifert genus must equal KnotInfo's three_genus. (Genera are standard values.)
GENUS_CASES = [
    ("3_1", 1),
    ("4_1", 1),
    ("5_1", 2),
    ("5_2", 1),
    ("6_1", 1),
    ("6_2", 2),
    ("6_3", 2),
    ("7_1", 3),
]


@pytest.mark.parametrize("name,genus", GENUS_CASES)
def test_seifert_genus_matches_known(name, genus):
    k = knots.from_name(name)
    s = seifert_structure(k.pd_code)
    assert s.genus == genus


def test_trefoil_structure():
    s = seifert_structure(knots.from_name("3_1").pd_code)
    assert s.seifert_circles == 2
    assert abs(s.writhe) == 3
    assert len(s.crossing_signs) == 3


def test_malformed_pd_raises():
    # Labels that do not follow the understrand convention.
    with pytest.raises(ValueError):
        seifert_structure(((1, 2, 4, 3), (3, 4, 2, 1)))
