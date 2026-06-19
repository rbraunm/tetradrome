# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Memory-bounded batch reduction (engine Phase 6).

The budgeted scheduler must do three things: preserve the answer (scheduling only changes
timing, not homology), keep co-resident memory within the budget, and fail loud when a single
complex cannot fit at any concurrency -- the agreement discipline extended to memory-aware
scheduling. The packing logic is a pure function, tested directly; answer-preservation and
fail-loud are tested through the real reducer.
"""
import pytest

from tetradrome.algebra import f2_homology, parallel_f2_homology, predict_size
from tetradrome.algebra.parallel import _pack_waves
from tetradrome.engines.floer import GridDiagram, grid_complexes

KNOT = "5_2"   # several Alexander gradings, small enough for the sandbox


def _complexes():
    return grid_complexes(GridDiagram.from_knotinfo(KNOT))


def test_pack_waves_respects_budget_and_worker_cap():
    priced = [(40, "a", None), (40, "b", None), (40, "c", None), (50, "d", None)]
    waves = _pack_waves(priced, workers=2, budget=100)
    for wave in waves:
        assert len(wave) <= 2
        assert sum(peak for peak, _, _ in wave) <= 100
    placed = sorted(key for wave in waves for _, key, _ in wave)
    assert placed == ["a", "b", "c", "d"]


def test_pack_waves_isolates_an_item_that_only_fits_alone():
    priced = [(100, "big", None), (10, "x", None), (10, "y", None)]
    waves = _pack_waves(priced, workers=8, budget=100)
    big_wave = next(wave for wave in waves if any(key == "big" for _, key, _ in wave))
    assert big_wave == [(100, "big", None)]


def test_budgeted_reduction_matches_serial_reference():
    complexes = _complexes()
    reference = {a: f2_homology(cx) for a, cx in complexes.items()}
    generous = parallel_f2_homology(complexes, workers=3, ram_budget_bytes=10**12)
    assert generous == reference


def test_tight_budget_preserves_answer():
    # Budget = the single largest grading's peak, so the scheduler cannot co-resident the heavy
    # gradings and must serialize into multiple waves -- the answer must still be identical.
    complexes = _complexes()
    reference = {a: f2_homology(cx) for a, cx in complexes.items()}
    budget = max(predict_size(cx).packed_peak_bytes for cx in complexes.values())
    tight = parallel_f2_homology(complexes, workers=8, ram_budget_bytes=budget)
    assert tight == reference


def test_complex_over_budget_fails_loud():
    complexes = _complexes()
    smallest = min(predict_size(cx).packed_peak_bytes for cx in complexes.values())
    with pytest.raises(MemoryError):
        parallel_f2_homology(complexes, workers=4, ram_budget_bytes=smallest - 1)
