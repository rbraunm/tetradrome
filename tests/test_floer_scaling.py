# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Agreement tests for the grid engine's accelerated paths (engine Phase 6 / Phase 5 tiers).

The whole acceleration argument rests on identical answers: switching reducer backend,
reducing the independent gradings across processes, or generating the complexes across
processes must all reproduce the serial reference result exactly. Also covers the synthetic
staircase grid used to push generator counts in the scaling study.
"""
from collections import defaultdict

import pytest

from tetradrome.algebra import predict_size
from tetradrome.engines.floer import (
    GridDiagram,
    grid_complexes,
    grid_poincare,
    hfk_hat,
    parallel_grid_complexes,
    reduce_complexes,
    staircase_grid,
)
from tetradrome.engines.floer.scaling import dense_reduction_bytes, grading_histogram

AGREEMENT_KNOTS = ["3_1", "4_1", "5_2"]


@pytest.mark.parametrize("name", AGREEMENT_KNOTS)
def test_backends_agree_with_reference(name):
    grid = GridDiagram.from_knotinfo(name)
    reference = grid_poincare(grid, backend="reference")
    assert grid_poincare(grid, backend="bitint") == reference


@pytest.mark.parametrize("name", AGREEMENT_KNOTS)
def test_parallel_reduction_agrees(name):
    grid = GridDiagram.from_knotinfo(name)
    reference = grid_poincare(grid, backend="reference", workers=1)
    assert grid_poincare(grid, backend="bitint", workers=3) == reference


@pytest.mark.parametrize("name", AGREEMENT_KNOTS)
def test_parallel_generation_matches_serial(name):
    # Stronger than answer-equality: parallel generation must reproduce the serial reference
    # complex bit-for-bit. Position assignments within each (Alexander, degree) block depend on
    # enumeration order, so this pins that contiguous lexicographic slicing reproduces
    # itertools.permutations exactly -- what makes parallel generation an answer-preserving
    # acceleration (ADR 0011 type B, the reference run in parallel), not a separate path.
    grid = GridDiagram.from_knotinfo(name)
    serial = grid_complexes(grid)
    parallel = parallel_grid_complexes(grid, workers=3)
    assert parallel.keys() == serial.keys()
    for a_grading in serial:
        assert parallel[a_grading].degrees() == serial[a_grading].degrees()
        for degree in serial[a_grading].degrees():
            assert parallel[a_grading].dim(degree) == serial[a_grading].dim(degree)
            assert parallel[a_grading].differential(degree) == serial[a_grading].differential(degree)


def test_staircase_grid_is_valid_and_trivial():
    grid = staircase_grid(5)
    assert grid.n == 5
    assert sorted(grid.O) == list(range(5)) and sorted(grid.X) == list(range(5))
    # the unknot: a single generator in HFK-hat at the origin
    assert hfk_hat(grid) == {(0, 0): 1}


@pytest.mark.parametrize("name", AGREEMENT_KNOTS)
def test_histogram_cost_matches_built_complex_cost(name):
    # The build-free predictor (grading_histogram -> dense_reduction_bytes) must agree with the
    # built-complex predictor (predict_size): same per-block cost, same dims. That agreement is
    # what lets the memory guard price a size it never builds. A degree-convention drift or a
    # formula drift between the two would break this exactly.
    grid = GridDiagram.from_knotinfo(name)
    complexes = grid_complexes(grid)
    histogram = grading_histogram(grid)

    by_alexander = defaultdict(dict)
    for (a_grading, degree), count in histogram.items():
        by_alexander[a_grading][degree] = count
    assert set(by_alexander) == set(complexes)
    for a_grading, cx in complexes.items():
        assert by_alexander[a_grading] == {n: cx.dim(n) for n in cx.degrees()}

    built = sum(predict_size(cx).packed_peak_bytes for cx in complexes.values())
    assert dense_reduction_bytes(histogram) == built
