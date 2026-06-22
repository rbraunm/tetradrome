# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The accelerator axis: everything GPU-vendor-specific the scheduler needs, one class per vendor.

``Accelerator`` owns the device-vendor mechanics -- enumerating the devices and their VRAM, and
(in the warm worker) standing up and tearing down a device session. ``detect_accelerator()``
returns the implementation for whatever runtime is actually present: an NVIDIA CUDA device that
cupy can drive returns ``NvidiaCudaAccelerator``; a different accelerator that is detected but
unsupported (an AMD ROCm runtime, today) raises loudly the same way ``for_host`` does for an
unknown OS, naming what to build; nothing present returns None, meaning CPU-only, which is not an
error. Adding a vendor is one new subclass plus a branch in ``detect_accelerator``.

This keeps the vendor assumptions out of the rest of the scheduler: the inventory enumerates
devices through this axis, and a non-NVIDIA box is either supported or fails loud, never silently
reported as having no GPU.
"""
from __future__ import annotations

import abc
import os
import shutil

from .inventory import GPU


class Accelerator(abc.ABC):
    """All GPU-vendor-specific behavior the scheduler needs, in one place per vendor. The layers
    above consume the enumerated devices and the session hooks without knowing the vendor."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A short identifier for logs, e.g. 'nvidia-cuda'."""

    @abc.abstractmethod
    def detect_devices(self) -> tuple[GPU, ...]:
        """The usable devices on this box with their VRAM, empty if none are usable."""


class NvidiaCudaAccelerator(Accelerator):
    """NVIDIA CUDA devices, driven through cupy. The one implemented and validated vendor. Device
    enumeration goes through the hardware detector under algebra, imported lazily so this module
    stays independent of the math layers."""

    @property
    def name(self) -> str:
        return "nvidia-cuda"

    def detect_devices(self) -> tuple[GPU, ...]:
        from ..algebra import gpu

        info = gpu.detect_gpu()
        if not info.device_usable:
            return ()
        return tuple(
            GPU(index=i, vram_bytes=(dev.memory_total_mib or 0) * 1024 * 1024, numa_node=None)
            for i, dev in enumerate(info.devices))


# ---- warm-worker session hooks (NVIDIA/cupy) ----------------------------
# Module-level so they pickle to a spawned warm worker by reference. They run inside that worker,
# which the executor has already pointed at the device via CUDA_VISIBLE_DEVICES.

def _cupy():
    import cupy
    return cupy


def gpu_session_setup() -> None:
    """Create and warm the CUDA context once, before the first job, so the first job's timing is
    compute and not the one-time cost of standing the context up."""
    cupy = _cupy()
    cupy.cuda.Device().use()
    warm = cupy.zeros(1, dtype=cupy.uint64)
    warm += 1                                   # forces context creation and runtime warmup
    cupy.cuda.Stream.null.synchronize()


def gpu_session_between() -> None:
    """Free the cupy memory pools after a job so the next small job starts from a clean pool while
    the held context stays up."""
    cupy = _cupy()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


# ---- detection / dispatch -----------------------------------------------

def _rocm_present() -> bool:
    """Whether an AMD ROCm runtime looks installed. Used only to fail loud when an accelerator we
    do not yet support is present, rather than silently reporting no GPU."""
    return bool(shutil.which("rocminfo") or shutil.which("rocm-smi")
                or os.path.isdir("/opt/rocm"))


def detect_accelerator() -> Accelerator | None:
    """The accelerator for whatever runtime is present, or None for CPU-only. Raises loudly when a
    device is present that no Accelerator supports yet, so that vendor support is a deliberate
    addition rather than a box silently running CPU-only."""
    from ..algebra import gpu

    if gpu.detect_gpu().device_usable:
        return NvidiaCudaAccelerator()
    if _rocm_present():
        raise NotImplementedError(
            "an AMD ROCm runtime is present but no Accelerator supports it yet; implement an "
            "AmdRocmAccelerator and add a branch in detect_accelerator().")
    return None
