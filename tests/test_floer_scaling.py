# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Agreement tests for the grid engine's accelerated paths (engine Phase 6 / Phase 5 tiers).

The whole acceleration argument rests on identical answers: switching reducer backend,
reducing the independent gradings across processes, or generating the complexes across
processes must all reproduce the serial reference result exactly. Also covers the synthetic
staircase grid used to push generator counts in the scaling study.
"""
import pytest

from tetradrome.engines.floer import (
    GridDiagram,
    grid_complexes,
    grid_poincare,
    hfk_hat,
    parallel_grid_complexes,
    reduce_complexes,
    staircase_grid,
)

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
