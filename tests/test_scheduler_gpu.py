# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""End-to-end GPU reductions through the scheduler, on a real device.

This is the one part of the scheduler that needs hardware: the warm worker holding a real CUDA
context and the packed-gpu reducer running through both execution paths. The whole module skips
when cupy is not usable, so it is green in the sandbox and CI and runs for real on a GPU host.

Each path is forced by the routing knob. vram_fraction 1.0 keeps reductions in the single serial
warm worker, so the full grid is safe: one context, one job at a time. vram_fraction 0.0 forces a
fresh process per reduction, so the fresh test uses only TWO reductions on purpose. Forcing fresh
on a full grid spawns a CUDA context per reduction all at once, and on a desktop whose GPU also
drives the display that swarm exhausts the card and hangs the machine. Never force fresh on more
than a couple of jobs; real routing never does this, since the router sends small jobs warm and
only large, VRAM-bound jobs fresh.
"""
import pytest
from collections import defaultdict

from tetradrome.algebra import available_f2_backends, f2_homology
from tetradrome.algebra.gpu import usable_cupy
from tetradrome.engines.floer.generation import grid_complexes
from tetradrome.engines.floer.grid import staircase_grid
from tetradrome.engines.floer.scheduling import reduction_graph
from tetradrome.scheduler import Placement, Scheduler, detect_machine

pytestmark = pytest.mark.skipif(not usable_cupy(), reason="no usable CUDA device / cupy")


def _packed_gpu_available() -> bool:
    return any(name == "packed-gpu" and ok for name, ok, _note in available_f2_backends())


def _assembled_oracle(complexes):
    # What the assembly produces from bitint reductions: dimensions summed into (-degree, alexander),
    # zeros dropped. The scheduled GPU run's terminal must equal this. The per-grading reductions
    # are intermediate now (the assembly consumes them) and freed once it is dispatched, so the
    # terminal is what we check.
    poincare = defaultdict(int)
    for alexander, cx in complexes.items():
        for degree, dimension in f2_homology(cx, "bitint").items():
            poincare[(-degree, alexander)] += dimension
    return {key: value for key, value in poincare.items() if value}


def test_gpu_warm_path_matches_oracle():
    # vram_fraction 1.0 keeps every reduction in the one serial warm worker: a single held
    # context, one job at a time, so the full grid is safe to run on a desktop GPU.
    assert _packed_gpu_available(), "cupy is usable but packed-gpu is not listed as a backend"
    complexes = grid_complexes(staircase_grid(7))
    graph, assemble_key = reduction_graph(complexes, backend="auto")
    report = Scheduler(detect_machine(), vram_fraction=1.0).run(graph)
    assert not report.failures, report.failures
    assert report.results[assemble_key] == _assembled_oracle(complexes)
    assert report.calibration.rate(Placement.GPU) is not None     # calibrated from real runs


def test_gpu_fresh_path_matches_oracle():
    # vram_fraction 0.0 forces a fresh process per reduction. DELIBERATELY two reductions only, so
    # at most two fresh CUDA contexts are ever resident. See the module docstring: forcing fresh
    # on a full grid swarms the GPU and hangs a desktop. Two proves the fresh path is correct.
    assert _packed_gpu_available(), "cupy is usable but packed-gpu is not listed as a backend"
    complexes = grid_complexes(staircase_grid(7))
    pair = dict(list(complexes.items())[:2])
    graph, assemble_key = reduction_graph(pair, backend="auto")
    report = Scheduler(detect_machine(), vram_fraction=0.0).run(graph)
    assert not report.failures, report.failures
    assert report.results[assemble_key] == _assembled_oracle(pair)
