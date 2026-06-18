# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

import pytest

from tetradrome import knots
from tetradrome.errors import UnknownKnot


@pytest.mark.parametrize(
    "given,identity,crossings",
    [
        ("3_1", "3_1", 3),
        ("4_1", "4_1", 4),
        ("K11n34", "11n_34", 11),  # Spherogram-style name normalizes to KnotInfo's
        ("11n34", "11n_34", 11),
    ],
)
def test_from_name_via_knotinfo(given, identity, crossings):
    k = knots.from_name(given)
    assert k.identity == identity
    assert k.source_notation == "name"
    assert k.crossing_number == crossings
    assert all(len(entry) == 4 for entry in k.pd_code)


def test_from_name_unknown_raises():
    with pytest.raises(UnknownKnot):
        knots.from_name("not_a_real_knot_zzz")


def test_from_pd_round_trips():
    k = knots.from_name("4_1")
    again = knots.from_pd(k.pd_code, identity="4_1")
    assert again.pd_code == k.pd_code
    assert again.source_notation == "pd"
    assert again.identity == "4_1"


def test_normalized_diagram_is_frozen():
    k = knots.from_name("3_1")
    with pytest.raises(Exception):
        k.pd_code = ()  # type: ignore[misc]
