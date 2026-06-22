# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""GPU session hooks for the warm worker: hold one CUDA context across jobs and free the pool
between them.

The warm worker takes a setup callable it runs once before its first job and a between callable
it runs after every job. On a real device, setup creates and warms the context so the first job's
measured time is compute rather than context creation, and between frees the cupy memory pools so
each small job starts from a clean pool while the context itself stays up. Both lazy-import cupy,
so the package imports fine on a host without it; called without cupy they raise rather than
pretend to work.

Pass them to the scheduler as ``Scheduler(machine, warm_setup=gpu_session_setup,
warm_between=gpu_session_between)``. They run inside the warm worker process, which the executor
has already pinned to the device via CUDA_VISIBLE_DEVICES, so the visible device is the right one.
"""
from __future__ import annotations


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
    """Free the cupy memory pools after a job so the next small job starts from a clean pool
    while the held context stays up."""
    cupy = _cupy()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
