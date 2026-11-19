# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The native Floer invariants through compute() (checkpoint 4b).

The chirality claim is load-bearing: KnotInfo's HFK/tau/genus columns share its PD
chirality and the native grid is built in that standard chirality, so the native value
must equal the RAW KnotInfo oracle -- no mirror, no transpose. The tier-0 set includes
chiral knots (nonzero tau), so a stray sign flip or reflection cannot pass. Under
strict, every result here is cross-checked by a computed oracle (kfh) AND KnotInfo.
"""
import pytest

from tetradrome import invariants, knots
from tetradrome.backends import knotinfo_backend as ki
from tetradrome.diagrams.reduction import reduced
from tetradrome.engines.floer import grid_store, hfk_hat

TIER0 = ["3_1", "4_1", "8_19"]
FLOER = ["knot_floer_homology", "ozsvath_szabo_tau", "three_genus"]


# --- the mirror check: native equals the RAW oracle on chiral knots ---

@pytest.mark.parametrize("name", TIER0)
@pytest.mark.parametrize("invariant", FLOER)
def test_native_floer_matches_knotinfo_raw_under_strict(invariant, name):
    result = invariants.compute(knots.from_name(name), invariant)  # strict default
    assert result.value == ki.known_answer(name, invariant)
    # strict passed, so a computed oracle agreed; both validators are on record.
    assert result.validation.verdict("knot_floer_homology") == "pass"
    assert result.validation.verdict("knotinfo") == "pass"
    assert result.validation.is_validated
    assert not result.validation.has_disagreement


def test_known_scalar_floer_values():
    # 8_19 = T(3,4): tau = 3, genus = 3 in KnotInfo's raw chirality; 3_1 genus = 1
    # (genus is chirality-free).
    assert invariants.compute(knots.from_name("8_19"), "ozsvath_szabo_tau").value == 3
    assert invariants.compute(knots.from_name("8_19"), "three_genus").value == 3
    assert invariants.compute(knots.from_name("3_1"), "three_genus").value == 1


# --- provenance of the native path ---

def test_floer_provenance_records_the_libraries_its_kernels_run_on():
    result = invariants.compute(knots.from_name("3_1"), "knot_floer_homology")
    assert result.provenance.backend == "tetradrome-native"
    assert result.provenance.method == "grid_hfk_hat"
    assert result.provenance.inputs == "pd_code"
    # The grid kernels run on numpy, compiled by Numba when it is installed.
    from tetradrome.engines.floer.differential_jit import HAVE_NUMBA

    expected = ["python", "numpy", "numba"] if HAVE_NUMBA else ["python", "numpy"]
    assert [name for name, _ in result.provenance.library_versions] == expected
    assert result.validation.d_squared_check == "not_applicable"


def test_floer_method_labels_per_invariant():
    tau = invariants.compute(knots.from_name("3_1"), "ozsvath_szabo_tau")
    genus = invariants.compute(knots.from_name("3_1"), "three_genus")
    assert tau.provenance.method == "grid_filtered_tau"
    assert genus.provenance.method == "grid_hfk_genus"


# --- modes on the Floer path ---

def test_floer_off_returns_full_provenance_without_validators():
    result = invariants.compute(
        knots.from_name("3_1"), "knot_floer_homology", validate="off"
    )
    assert result.validation.validators == ()
    assert result.value == ki.known_answer("3_1", "knot_floer_homology")
    assert result.provenance.method == "grid_hfk_hat"


def test_floer_provenance_records_the_grid_and_where_it_came_from():
    result = invariants.compute(knots.from_name("5_2"), "knot_floer_homology", validate="off")
    grid = result.provenance.grid
    assert grid.diagram == reduced(knots.from_name("5_2").pd_code)
    assert grid.size == 7
    assert grid.source in ("built", "stored")
    assert grid.generator == grid_store.generator_stamp()
    assert result.value == hfk_hat(grid.grid)
    again = invariants.compute(knots.from_name("5_2"), "knot_floer_homology", validate="off")
    assert again.provenance.grid.source == "stored"
    assert again.provenance.grid.built_at == grid.built_at
    assert again.provenance.grid.o_markers == grid.o_markers


# --- any knot: a PD code without a name, or a braid ---

def test_floer_of_an_unnamed_pd_code_is_validated_by_the_computed_oracle():
    raw = knots.from_pd(knots.from_name("3_1").pd_code)  # identity is None
    result = invariants.compute(raw, "knot_floer_homology")
    assert result.value == ki.known_answer("3_1", "knot_floer_homology")
    assert result.validation.verdict("knot_floer_homology") == "pass"


@pytest.mark.parametrize("name", ["3_1", "5_2", "8_19"])
def test_floer_of_a_braid_is_the_knot_s_with_its_chirality(name):
    braid = knots.from_braid(ki.braid_word(name))
    for invariant in ("knot_floer_homology", "ozsvath_szabo_tau"):
        result = invariants.compute(braid, invariant, validate="off")
        assert result.value == ki.known_answer(name, invariant)
        assert result.provenance.inputs == "braid_word"
