import pytest

from tetradrome.backends import knotinfo_backend as ki
from tetradrome.errors import UnknownKnot


@pytest.mark.parametrize(
    "given,expected",
    [
        ("3_1", "3_1"),
        ("10_124", "10_124"),
        ("K11n34", "11n_34"),
        ("11n34", "11n_34"),
        ("11a12", "11a_12"),
    ],
)
def test_normalize_name(given, expected):
    assert ki.normalize_name(given) == expected


def test_lookup_unknown_raises():
    with pytest.raises(UnknownKnot):
        ki.lookup("definitely_not_a_knot")


def test_braid_word_and_oracle_present():
    assert ki.braid_word("3_1") == [1, 1, 1]
    assert ki.braid_word("4_1") == [1, -2, 1, -2]
    assert ki.known_answer("6_2", "determinant") == 11
    assert ki.known_answer("6_2", "signature") == -2


def test_known_answer_unsupported_invariant_is_none():
    assert ki.known_answer("3_1", "jones_polynomial") is None
