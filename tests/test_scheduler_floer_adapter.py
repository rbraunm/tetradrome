# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The scheduled reduction must equal the in-process reduction, for every backend and knot.

This pins the adapter against an independent oracle: the same per-grading f2_homology folded
into the same Poincare count, computed directly in this process. If the scheduler's spawn,
dependency passing, and assembly change any answer, this fails.
"""
import sys
from collections import defaultdict

import pytest

from tetradrome.algebra import available_f2_backends, f2_homology, predict_cost
from tetradrome.engines.floer.generation import grading_histogram_reference, grid_complexes
from tetradrome.engines.floer.differential_jit import HAVE_NUMBA
from tetradrome.engines.floer.grid import staircase_grid
from tetradrome.scheduler import ComputePath, Job, JobGraph, Placement, Scheduler, detect_machine
from tetradrome.engines.floer.scheduling import (
    generation_graph,
    histogram_graph,
    reduction_graph,
    reduction_jobs,
    whole_knot_graph,
)

_BACKENDS = [name for name, ok, _ in available_f2_backends() if ok and name != "packed-gpu"]


def _inprocess(complexes, backend):
    poincare = defaultdict(int)
    for alexander, cx in complexes.items():
        for degree, dimension in f2_homology(cx, backend).items():
            poincare[(-degree, alexander)] += dimension
    return {key: value for key, value in poincare.items() if value}


def _scheduled(complexes, backend):
    graph, assemble_key = reduction_graph(complexes, backend=backend)
    report = Scheduler(detect_machine()).run(graph)
    assert report.failures == []
    return report.results[assemble_key]


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("backend", _BACKENDS)
def test_scheduled_reduction_matches_in_process(n, backend):
    complexes = grid_complexes(staircase_grid(n))
    assert _scheduled(complexes, backend) == _inprocess(complexes, backend)


def test_all_backends_agree_through_the_scheduler():
    complexes = grid_complexes(staircase_grid(5))
    answers = {backend: _scheduled(complexes, backend) for backend in _BACKENDS}
    reference = answers["bitint"]
    for backend, answer in answers.items():
        assert answer == reference, f"{backend} disagrees with bitint through the scheduler"


def test_reduction_jobs_carry_predicted_cost():
    # each reduction job is priced with predict_cost of its complex; the assembly merge is free
    complexes = grid_complexes(staircase_grid(5))
    jobs, assemble_key = reduction_jobs(complexes, backend="bitint")
    by_key = {job.key: job for job in jobs}
    for alexander, cx in complexes.items():
        assert by_key[("reduce", alexander)].cost == predict_cost(cx)
    assert by_key[assemble_key].cost == 0.0


def test_generation_graph_matches_serial():
    # Generation through the scheduler must reproduce the serial reference complex bit-for-bit, not
    # merely agree on homology: positions within each (Alexander, degree) block depend on
    # enumeration order, so this pins that the contiguous lexicographic slices, folded in
    # slice-index order across spawned workers, reproduce grid_complexes exactly. slice_states is
    # forced small so a tiny grid still splits into several slices and the merge ordering is tested.
    grid = staircase_grid(5)                                  # 120 states
    graph, merge_key = generation_graph(grid, slice_states=32)
    report = Scheduler(detect_machine()).run(graph)
    assert report.failures == []
    produced = report.results[merge_key]
    serial = grid_complexes(grid)
    assert produced.keys() == serial.keys()
    for a_grading in serial:
        assert produced[a_grading].degrees() == serial[a_grading].degrees()
        for degree in serial[a_grading].degrees():
            assert produced[a_grading].dim(degree) == serial[a_grading].dim(degree)
            assert produced[a_grading].differential(degree) == serial[a_grading].differential(degree)


def _bound_kernels(inputs, deps):
    from tetradrome.engines.floer import differential_jit, gradings_jit

    return (gradings_jit._kernel is not None, differential_jit._kernel is not None,
            differential_jit._rank_kernel is not None, differential_jit._unrank_kernel is not None)


@pytest.mark.skipif(sys.platform == "win32" or not HAVE_NUMBA,
                    reason="only a forkserver can hand its workers bound numba kernels")
def test_workers_start_with_the_grid_kernels_bound():
    # A worker that had to bind the kernels itself would pay about 0.6 s before any work; the
    # forkserver binds them once (engines/floer/prewarm.py) and every fork inherits them.
    probe = Job(key="probe", run=_bound_kernels, inputs={},
                paths=(ComputePath(Placement.CPU_PINNED, cores=1, ram_bytes=1 << 20),))
    report = Scheduler(detect_machine()).run(JobGraph([probe]))
    assert report.failures == []
    assert report.results["probe"] == (True, True, True, True)


def test_histogram_graph_matches_serial():
    # The scheduled histogram counts each slice with the block grading kernel and sums the slices;
    # the reference walks every state in pure Python. slice_states is forced small so a tiny grid
    # still splits into several slices, one of them uneven (120 states over four slices of 30).
    grid = staircase_grid(5)
    graph, merge_key = histogram_graph(grid, slice_states=32)
    report = Scheduler(detect_machine()).run(graph)
    assert report.failures == []
    assert len(graph) == 5
    assert report.results[merge_key] == grading_histogram_reference(grid)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_whole_knot_graph_matches_in_process(backend):
    # The end-to-end graph -- generation slices, a merge partitioned by Alexander grading, a reduce
    # per grading reading only its shard, and the assembly -- must equal the in-process Poincare
    # count: the same per-grading f2_homology folded the same way, now through the scheduler's
    # spawn, partitioned merge, shard routing, and assembly.
    grid = staircase_grid(5)
    graph, assemble_key = whole_knot_graph(grid, grading_histogram_reference(grid), backend=backend)
    report = Scheduler(detect_machine()).run(graph)
    assert report.failures == []
    assert report.results[assemble_key] == _inprocess(grid_complexes(grid), backend)
