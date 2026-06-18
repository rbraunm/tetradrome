# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for GPU detection, auto-configuration, and enablement guidance.

The probes run against whatever hardware the host actually has: they must never raise,
and their verdict must match the host's ground truth (a CUDA device established directly,
not via the probe). The decision logic -- which config a card produces, and what guidance
each detection state yields -- is exercised against constructed GpuInfo objects so every
branch is covered without a device.
"""
from tetradrome.algebra import gpu
from tetradrome.algebra.gpu import GpuDevice, GpuInfo


def _info(**kw):
    base = dict(driver_present=False, driver_version=None, driver_cuda_version=None,
                cupy_installed=False, device_usable=False, devices=(), detail=None)
    base.update(kw)
    return GpuInfo(**base)


def _cuda_device_present():
    """Host ground truth, established directly rather than via the probe under test."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def test_detection_matches_host_hardware():
    info = gpu.detect_gpu()                 # never raises, with or without a GPU
    has_cuda = _cuda_device_present()
    assert info.device_usable is has_cuda
    assert isinstance(gpu.format_report(info), str)
    if has_cuda:
        import cupy
        assert gpu.usable_cupy() is cupy
    else:
        assert gpu.usable_cupy() is None


def test_wheel_mapping():
    assert gpu._cupy_wheel("13.1") == "cupy-cuda13x"
    assert gpu._cupy_wheel("12.4") == "cupy-cuda12x"
    assert gpu._cupy_wheel("11.8") == "cupy-cuda11x"
    assert gpu._cupy_wheel("99.0") is None      # unknown line -> no guess
    assert gpu._cupy_wheel(None) is None


def test_config_disabled_when_not_usable():
    cfg = gpu.gpu_config(_info())
    assert cfg.enabled is False and cfg.vram_budget_mib is None


def test_config_picks_most_free_device_and_budgets_vram():
    info = _info(
        device_usable=True,
        devices=(
            GpuDevice("card-a", "7.5", 8192, 2000),
            GpuDevice("card-b", "8.6", 24576, 20000),
        ),
    )
    cfg = gpu.gpu_config(info)
    assert cfg.enabled and cfg.device_id == 1               # card-b has more free
    assert cfg.compute_capability == "8.6"
    assert cfg.vram_budget_mib == int(20000 * 0.8)


def test_instructions_none_when_usable_or_no_hardware():
    assert gpu.enablement_instructions(_info(device_usable=True)) is None
    assert gpu.enablement_instructions(_info()) is None     # no driver, no devices


def test_instructions_driver_present_cupy_missing_known_cuda():
    info = _info(driver_present=True, driver_cuda_version="12.4",
                 devices=(GpuDevice("some-nvidia-gpu", "8.9", 16384, 16000),))
    steps = gpu.enablement_instructions(info)
    assert "cupy-cuda12x" in steps and "some-nvidia-gpu" in steps


def test_instructions_driver_present_cupy_missing_unknown_cuda():
    info = _info(driver_present=True, driver_cuda_version=None,
                 devices=(GpuDevice("some-nvidia-gpu", None, None, None),))
    steps = gpu.enablement_instructions(info)
    assert "nvidia-smi" in steps                            # falls back to guidance, no guess


def test_instructions_cupy_installed_but_unusable():
    info = _info(driver_present=True, cupy_installed=True,
                 detail="cupy is installed but no CUDA device is visible")
    steps = gpu.enablement_instructions(info)
    assert "match" in steps.lower()


def test_instructions_hardware_but_no_driver():
    info = _info(devices=(GpuDevice("some-nvidia-gpu", None, None, None),))
    steps = gpu.enablement_instructions(info)
    assert "driver" in steps.lower()
