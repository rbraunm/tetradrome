# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for complex-size prediction and backend routing (engine Phase 5).

predict_size reports exact structural facts, so it is checked against the actual complex.
route_backend is a decision function, so each branch is exercised with constructed sizes
and GpuConfig objects -- no device needed -- to pin down exactly when work goes to the GPU,
when it stays on the CPU, and when an over-budget prediction must fail loudly.
"""
import pytest

from tetradrome import knots
from tetradrome.algebra import tiers
from tetradrome.algebra.gpu import GpuConfig
from tetradrome.algebra.memory import (
    ComplexSize,
    dense_block_ops,
    predict_cost,
    predict_size,
    route_backend,
)
from tetradrome.engines import khovanov

# A complex whose largest grading is known, to check the peak arithmetic exactly.
from tetradrome.algebra import GradedComplex


def _available(gpu_ok: bool):
    return [
        ("reference", True, ""),
        ("bitint", True, ""),
        ("packed-cpu", True, ""),
        ("packed-gpu", gpu_ok, ""),
    ]


def test_predict_size_matches_actuals():
    pd = knots.from_name("6_2").pd_code
    for cx in khovanov.khovanov_complexes(pd).values():
        size = predict_size(cx)
        assert size.total_generators == cx.total_dim()
        assert size.dims == {n: cx.dim(n) for n in cx.degrees()}
        n, cols, rows = size.largest_map
        if size.packed_peak_bytes:
            assert cols == cx.dim(n) and rows == cx.dim(n + 1)


def test_predict_size_peak_arithmetic():
    # one map C^0 (3 cols) -> C^1 (2 rows): nwords=1, peak=(3+min(3,2))*1*8 = 40 bytes
    cx = GradedComplex({0: 3, 1: 2}, {0: [{0}, {1}, {0}]})
    size = predict_size(cx)
    assert size.largest_map == (0, 3, 2)
    assert size.packed_peak_bytes == (3 + 2) * 1 * 8


def test_dense_block_ops_arithmetic():
    # 3 cols into 2 rows: nwords=1, ops = cols*min(cols,rows)*nwords = 3*2*1 = 6
    assert dense_block_ops(3, 2) == 6
    # 100 cols into 70 rows: nwords=ceil(70/64)=2, ops = 100*70*2 = 14000
    assert dense_block_ops(100, 70) == 14000
    # an empty side is no work
    assert dense_block_ops(0, 5) == 0
    assert dense_block_ops(5, 0) == 0


def test_predict_cost_sums_block_ops():
    # C^0 (3) -> C^1 (2) -> C^2 (0): only the first block costs, 3*2*1 = 6
    cx = GradedComplex({0: 3, 1: 2}, {0: [{0}, {1}, {0}]})
    assert predict_cost(cx) == dense_block_ops(3, 2)


def test_predict_cost_grows_with_complex_size():
    small = predict_cost(GradedComplex({0: 4, 1: 4}, {0: [{0}, {1}, {2}, {3}]}))
    big = predict_cost(GradedComplex({0: 40, 1: 40}, {0: [{i} for i in range(40)]}))
    assert big > small > 0


def test_predict_cost_is_dimension_based_like_predict_size():
    # predict_cost predicts the dense block from the dimensions (an upper bound), the same basis
    # as predict_size: C^0=5 -> C^1=3 predicts 5*3*1 ops even with the differential unpopulated.
    cx = GradedComplex({0: 5, 1: 3}, {})
    assert predict_cost(cx) == dense_block_ops(5, 3)


def test_predict_cost_zero_without_an_adjacent_degree():
    # a lone nonempty degree has no boundary block to reduce
    assert predict_cost(GradedComplex({0: 5}, {})) == 0


_GPU_ON = GpuConfig(enabled=True, device_id=0, vram_budget_mib=1024, compute_capability="8.6")
_GPU_OFF = GpuConfig(enabled=False, device_id=None, vram_budget_mib=None, compute_capability=None)


def _size(bytes_):
    return ComplexSize((0, 1), {0: 1, 1: 1}, 2, (0, 1, 1), bytes_)


def test_routes_cpu_when_no_gpu():
    d = route_backend(_size(10**9), _available(False), gpu_cfg=_GPU_OFF)
    assert d.backend == "bitint" and "no usable GPU" in d.reason


def test_routes_cpu_below_threshold():
    d = route_backend(_size(1 << 20), _available(True), gpu_cfg=_GPU_ON,
                      gpu_min_bytes=32 << 20)
    assert d.backend == "bitint" and "threshold" in d.reason


def test_routes_gpu_when_large_and_fits():
    d = route_backend(_size(64 << 20), _available(True), gpu_cfg=_GPU_ON,
                      gpu_min_bytes=32 << 20)
    assert d.backend == "packed-gpu" and d.fits_gpu


def test_routes_cpu_when_exceeds_vram():
    # 4 GiB needed, 1 GiB budget: above threshold but does not fit
    d = route_backend(_size(4 << 30), _available(True), gpu_cfg=_GPU_ON,
                      gpu_min_bytes=32 << 20)
    assert d.backend == "bitint" and "VRAM budget" in d.reason


def test_ram_budget_exceeded_raises():
    with pytest.raises(MemoryError):
        route_backend(_size(2 << 30), _available(False), gpu_cfg=_GPU_OFF,
                      ram_budget_bytes=1 << 30)


def test_route_accepts_a_complex_directly():
    cx = next(iter(khovanov.khovanov_complexes(knots.from_name("3_1").pd_code).values()))
    d = route_backend(cx, tiers.available_f2_backends())  # real detection, no GPU here
    assert d.backend in ("bitint", "reference")
