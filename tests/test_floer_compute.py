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

def test_grid_provenance_records_the_libraries_its_kernels_run_on():
    result = invariants.compute(knots.from_name("3_1"), "knot_floer_homology", engine="grid")
    assert result.provenance.backend == "tetradrome-native"
    assert result.provenance.method == "grid_hfk_hat"
    assert result.provenance.inputs == "pd_code"
    # The grid kernels run on numpy, compiled by Numba when it is installed.
    from tetradrome.engines.floer.differential_jit import HAVE_NUMBA

    expected = ["python", "numpy", "numba"] if HAVE_NUMBA else ["python", "numpy"]
    assert [name for name, _ in result.provenance.library_versions] == expected
    assert result.validation.d_squared_check == "not_applicable"


def test_floer_method_labels_per_invariant_and_engine():
    trefoil = knots.from_name("3_1")
    labels = {(invariant, engine): invariants.compute(trefoil, invariant, engine=engine)
              .provenance.method
              for invariant in FLOER for engine in ("auto", "grid")}
    assert labels == {
        ("knot_floer_homology", "auto"): "bordered_hfk_hat",
        ("three_genus", "auto"): "bordered_hfk_genus",
        ("ozsvath_szabo_tau", "auto"): "grid_filtered_tau",
        ("knot_floer_homology", "grid"): "grid_hfk_hat",
        ("three_genus", "grid"): "grid_hfk_genus",
        ("ozsvath_szabo_tau", "grid"): "grid_filtered_tau",
    }


# --- the engine choice ---

@pytest.mark.parametrize("name", ["3_1", "6_2", "8_19", "8_20"])
def test_both_engines_give_the_same_floer_values(name):
    knot = knots.from_name(name)
    for invariant in ("knot_floer_homology", "three_genus"):
        chosen = invariants.compute(knot, invariant, validate="off")
        named = invariants.compute(knot, invariant, validate="off", engine="grid")
        assert chosen.value == named.value == ki.known_answer(name, invariant)


def test_the_provenance_says_whether_the_engine_was_chosen_or_named():
    knot = knots.from_name("4_1")
    assert invariants.compute(knot, "knot_floer_homology").provenance.engine_choice == (
        "automatic")
    for engine in ("grid", "bordered"):
        result = invariants.compute(knot, "knot_floer_homology", engine=engine)
        assert result.provenance.engine_choice == "named"
        assert result.provenance.method == f"{engine}_hfk_hat"
    assert invariants.compute(knot, "determinant").provenance.engine_choice is None


def test_the_bordered_engine_records_its_morse_presentation():
    from tetradrome.diagrams.morse import morse_diagram

    pd = knots.from_name("7_4").pd_code
    result = invariants.compute(knots.from_name("7_4"), "knot_floer_homology", validate="off")
    source = result.provenance.presentation
    assert source.diagram == reduced(pd)
    assert source.morse == morse_diagram(reduced(pd))
    assert source.widest == max(source.morse.widths())
    assert [record.kind for record in result.provenance.computation.jobs] == [
        "invariant.bordered_hfk_hat"]


def test_naming_an_engine_that_cannot_answer_raises():
    with pytest.raises(ValueError, match="does not compute ozsvath_szabo_tau"):
        invariants.compute(knots.from_name("3_1"), "ozsvath_szabo_tau", engine="bordered")
    with pytest.raises(ValueError, match="has one engine"):
        invariants.compute(knots.from_name("3_1"), "jones_polynomial", engine="grid")
    with pytest.raises(ValueError, match="engine must be one of"):
        invariants.compute(knots.from_name("3_1"), "knot_floer_homology", engine="szabo")


# --- modes on the Floer path ---

def test_floer_off_returns_full_provenance_without_validators():
    result = invariants.compute(
        knots.from_name("3_1"), "knot_floer_homology", validate="off"
    )
    assert result.validation.validators == ()
    assert result.value == ki.known_answer("3_1", "knot_floer_homology")
    assert result.provenance.method == "bordered_hfk_hat"


def test_grid_provenance_records_the_grid_and_where_it_came_from():
    result = invariants.compute(knots.from_name("5_2"), "knot_floer_homology", validate="off",
                                engine="grid")
    grid = result.provenance.presentation
    assert grid.diagram == reduced(knots.from_name("5_2").pd_code)
    assert grid.size == 7
    assert grid.source in ("built", "stored")
    assert grid.generator == grid_store.generator_stamp()
    assert result.value == hfk_hat(grid.grid)
    again = invariants.compute(knots.from_name("5_2"), "knot_floer_homology", validate="off",
                               engine="grid")
    assert again.provenance.presentation.source == "stored"
    assert again.provenance.presentation.built_at == grid.built_at
    assert again.provenance.presentation.o_markers == grid.o_markers


def test_provenance_records_every_job_the_calculation_ran():
    from tetradrome.engines.floer import grading_histogram
    from tetradrome.engines.floer.scheduling import whole_knot_graph
    from tetradrome.scheduler import calibration_store

    result = invariants.compute(knots.from_name("6_2"), "knot_floer_homology", validate="off",
                                engine="grid")
    computation = result.provenance.computation
    grid = result.provenance.presentation.grid
    graph, _ = whole_knot_graph(grid, grading_histogram(grid))
    # The histogram run's jobs, then the calculation's: one record per job, none lost or doubled.
    assert len(computation.jobs) == len({record.job for record in computation.jobs})
    assert {repr(job.key) for job in graph.jobs()} <= {record.job for record in computation.jobs}
    kinds = {record.kind for record in computation.jobs}
    assert any(kind.startswith("f2.reduce.") for kind in kinds)
    assert {record.placement for record in computation.jobs} <= {
        "inline", "cpu_pinned", "cpu_unpinned", "gpu"}
    for record in computation.jobs:
        assert record.seconds >= 0
        assert (record.cpus == ()) == (record.placement == "inline")
    assert computation.source == calibration_store.software_identity()["source"]
    assert computation.machine["nodes"]


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
