# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tier-0 checks for the Szabo HFK-calculator validator (kfh), the KnotInfo three_genus accessor,
and the knot Floer known_answer dispatch. Real kfh runs here (n <= 8); the suite skips only if kfh
is not installed. These lock the plumbing and conventions the compute Floer wiring will rely on:
kfh and KnotInfo must agree raw, and the HFK ranks must be canonical-keyed (Maslov, Alexander).
"""
import pytest

pytest.importorskip("knot_floer_homology")

from tetradrome import knots
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.backends.hfk_adapter import HFKValidator

FLOER_INVARIANTS = ["knot_floer_homology", "ozsvath_szabo_tau", "three_genus"]
# Chiral tier-0 knots (nonzero tau), so a stray sign flip would be caught.
TIER0 = ["3_1", "4_1", "8_19"]


def test_validator_available_and_versioned():
    validator = HFKValidator()
    assert validator.is_available() is True
    info = validator.version_info()
    assert set(info) == {"knot_floer_homology"}
    assert info["knot_floer_homology"] not in ("", "absent")


@pytest.mark.parametrize("name", TIER0)
@pytest.mark.parametrize("invariant", FLOER_INVARIANTS)
def test_kfh_matches_knotinfo_raw(name, invariant):
    """kfh computed from our PD agrees with KnotInfo raw, under the canonical name and convention."""
    validator = HFKValidator()
    knot = knots.from_name(name)
    assert validator.known_value(knot, invariant) == ki.known_answer(name, invariant)


def test_hfk_ranks_keyed_maslov_alexander():
    """The HFK table is keyed (Maslov, Alexander) -- kfh's (Alexander, Maslov) must be transposed."""
    ranks = HFKValidator().known_value(knots.from_name("3_1"), "knot_floer_homology")
    assert ranks == {(-2, -1): 1, (-1, 0): 1, (0, 1): 1}
    assert ranks == ki.hfk_ranks("3_1")


def test_scalar_floer_values():
    validator = HFKValidator()
    assert validator.known_value(knots.from_name("8_19"), "ozsvath_szabo_tau") == 3
    assert validator.known_value(knots.from_name("8_19"), "three_genus") == 3


def test_out_of_scope_invariant_returns_none():
    assert HFKValidator().known_value(knots.from_name("3_1"), "determinant") is None


def test_no_pd_returns_none():
    braid_knot = knots.from_braid([1, 1, 1])  # braid presentation, no PD to run kfh on
    assert braid_knot.pd_code == ()
    assert HFKValidator().known_value(braid_knot, "ozsvath_szabo_tau") is None


def test_three_genus_accessor():
    assert ki.three_genus("3_1") == 1
    assert ki.three_genus("8_19") == 3


def test_known_answer_dispatches_floer_invariants():
    assert ki.known_answer("8_19", "ozsvath_szabo_tau") == 3
    assert ki.known_answer("8_19", "three_genus") == 3
    assert ki.known_answer("8_19", "knot_floer_homology") == ki.hfk_ranks("8_19")


def test_known_answer_unknown_invariant_is_none():
    assert ki.known_answer("3_1", "not_a_real_invariant") is None
