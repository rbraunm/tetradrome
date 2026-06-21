# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""End-to-end GPU reductions through the scheduler, on a real device.

This is the one part of the scheduler that needs hardware: the warm worker holding a real CUDA
context and the packed-gpu reducer running through both execution paths. The whole module skips
when cupy is not usable, so it is green in the sandbox and CI and runs for real on a GPU host.

It forces each path by the routing knob rather than by job size: vram_fraction 1.0 keeps small
reductions in the warm worker, vram_fraction 0.0 sends every one to a fresh process. In both, the
scheduled GPU result must equal the in-process pure-Python reduction for every grading.
"""
import pytest

from tetradrome.algebra import available_f2_backends, f2_homology
from tetradrome.algebra.gpu import usable_cupy
from tetradrome.engines.floer.generation import grid_complexes
from tetradrome.engines.floer.grid import staircase_grid
from tetradrome.engines.floer.scheduling import reduction_graph
from tetradrome.scheduler import Scheduler, detect_machine
from tetradrome.scheduler.gpu_session import gpu_session_setup, gpu_session_between

pytestmark = pytest.mark.skipif(not usable_cupy(), reason="no usable CUDA device / cupy")


def _packed_gpu_available() -> bool:
    return any(name == "packed-gpu" and ok for name, ok, _note in available_f2_backends())


@pytest.mark.parametrize("vram_fraction, path", [(1.0, "warm"), (0.0, "fresh")])
def test_gpu_reductions_match_pure_python_oracle(vram_fraction, path):
    assert _packed_gpu_available(), "cupy is usable but packed-gpu is not listed as a backend"
    complexes = grid_complexes(staircase_grid(7))
    oracle = {alexander: f2_homology(cx, "bitint") for alexander, cx in complexes.items()}

    graph, _assemble_key = reduction_graph(complexes, backend="auto")
    report = Scheduler(detect_machine(), vram_fraction=vram_fraction,
                       warm_setup=gpu_session_setup, warm_between=gpu_session_between).run(graph)

    assert not report.failures, report.failures
    for alexander, cx in complexes.items():
        assert report.results[("reduce", alexander)] == oracle[alexander]
    # the GPU placement was calibrated from real runs
    from tetradrome.scheduler import Placement
    assert report.calibration.rate(Placement.GPU) is not None
