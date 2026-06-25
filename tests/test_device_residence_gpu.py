# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm
"""Real-hardware integration test for device-resident results.

Skipped automatically unless cupy and a usable CUDA device are present, so it is a no-op in the
sandbox and in CI without a GPU, and runs on a real device (e.g. on the workstation). It validates
the end-to-end path the fabricated-device unit tests cannot: a producer makes a real cupy array
that stays resident in the warm worker's VRAM, and a warm consumer reads it on-device -- only a
scalar returns to the host, never the array.
"""
import pytest

from tetradrome.algebra.gpu import usable_cupy
from tetradrome.scheduler import (
    ComputePath,
    Job,
    JobGraph,
    Placement,
    Scheduler,
    detect_machine,
)

pytestmark = pytest.mark.skipif(usable_cupy() is None, reason="needs cupy and a usable CUDA GPU")

_N = 1 << 20                                 # 1,048,576 uint64 elements = 8 MiB on the device


def _gpu_path(vram_bytes):
    return ComputePath(Placement.GPU, cores=1, ram_bytes=1 << 20, vram_bytes=vram_bytes)


def make_resident_array(inputs, deps):
    import cupy as cp
    return cp.arange(_N, dtype=cp.uint64)    # the device buffer kept resident across jobs


def sum_on_device(inputs, deps):
    (resident,) = deps.values()               # the resident array, resolved from the registry
    return int(resident.sum().get())          # reduce on-device; only the scalar leaves the GPU


def test_real_device_resident_array_is_read_on_device():
    producer = Job(key="src", run=make_resident_array, inputs={}, paths=(_gpu_path(8 << 20),),
                   cost=1000, device_resident=True)
    consumer = Job(key="c", run=sum_on_device, inputs={}, paths=(_gpu_path(8 << 20),),
                   cost=1000, dependencies={"src"})
    report = Scheduler(detect_machine(), context_vram_reserve=0).run(JobGraph([producer, consumer]))
    assert report.failures == []
    assert report.device_count == 1                      # the result took the device-resident path
    assert "src" not in report.results                   # the array never came back to the host
    assert report.results["c"] == _N * (_N - 1) // 2     # sum 0..N-1, computed on the resident buffer
