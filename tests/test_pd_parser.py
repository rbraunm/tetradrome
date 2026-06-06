import pytest

from tetradrome.diagrams import pd


def test_normalize_accepts_valid_zero_based():
    raw = [(0, 1, 2, 3), (2, 3, 0, 1)]
    assert pd.normalize(raw) == ((0, 1, 2, 3), (2, 3, 0, 1))


def test_normalize_accepts_one_based():
    raw = [(1, 2, 3, 4), (3, 4, 1, 2)]
    assert pd.normalize(raw) == ((1, 2, 3, 4), (3, 4, 1, 2))


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        pd.normalize([])


def test_normalize_rejects_wrong_entry_length():
    with pytest.raises(ValueError):
        pd.normalize([(0, 1, 2)])


def test_normalize_rejects_label_not_appearing_twice():
    # 0 appears 3x, 2 appears 1x.
    with pytest.raises(ValueError):
        pd.normalize([(0, 0, 1, 2), (0, 1, 3, 3)])


def test_normalize_rejects_noncontiguous_labels():
    # All appear twice, but {0,1,2,9} is not a contiguous run of 4.
    with pytest.raises(ValueError):
        pd.normalize([(0, 1, 2, 9), (0, 1, 2, 9)])


def test_normalize_rejects_non_integer():
    with pytest.raises(ValueError):
        pd.normalize([("a", "b", "c", "d")])
