# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Execution routing for GPU jobs: warm held-context worker versus a fresh process.

A GPU job can run in a persistent worker that holds a CUDA context across jobs (fast dispatch,
but its cupy mempool is shared and must be freed between jobs) or in a fresh process (pays the
one-time cost of standing up a context, but reclaims all VRAM on exit). The choice is made per
job from two triggers:

  - vram footprint: a job whose predicted vram is a large fraction of the device budget goes
    fresh, so a big allocation lands in a process that frees it on exit rather than bloating the
    warm worker's pool. This is the firm trigger -- it keeps the warm worker healthy.
  - predicted time: a job whose predicted run time is large enough to amortize standing up a
    fresh context goes fresh too, since the context cost is then a small fraction of the work
    and the job gets clean VRAM. Below that, warm: a quick job should reuse the held context
    rather than pay to create a new one.

The predicted time comes from a job's op-count (predict_cost) times a per-placement rate the
Calibration learns from observed runtimes. Until a placement has been observed the rate is
unknown, the predicted time is None, and the time trigger is simply held off -- the decision is
vram-only -- until the first completed job of that placement seeds a real number.
"""
from __future__ import annotations

import enum


class Execution(enum.Enum):
    """How a GPU job runs: in the warm held-context worker, or in a fresh process."""
    WARM = "warm"
    FRESH = "fresh"


class Calibration:
    """Per-placement time-per-op estimates, refined from observed runtimes by an EWMA.

    ``predicted_time(cost, placement)`` is ``cost * rate[placement]`` once the placement has been
    observed, else None. ``observe`` folds each completed job's measured seconds into the rate
    for its placement: the per-op observation is ``seconds / cost``, blended with the running
    estimate. A job with no predicted cost (cost <= 0) carries no per-op signal and is skipped.

    State is per-run and in memory. Nothing is persisted: a stored rate would silently drift
    against changed hardware or backends, and a sweep of any size reaches a stable rate within
    its first few jobs anyway.
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._rate: dict = {}        # placement -> seconds per op, absent until observed

    def rate(self, placement):
        """The current seconds-per-op estimate for a placement, or None if never observed."""
        return self._rate.get(placement)

    def predicted_time(self, cost: float, placement):
        """Predicted run seconds for a job, or None if the placement has no rate yet."""
        rate = self._rate.get(placement)
        return None if rate is None else cost * rate

    def observe(self, cost: float, placement, seconds: float) -> None:
        """Fold a completed job's measured time into the rate for its placement."""
        if cost <= 0:
            return                   # no op-count, so no per-op signal to learn from
        observed = seconds / cost
        previous = self._rate.get(placement)
        if previous is None:
            self._rate[placement] = observed
        else:
            self._rate[placement] = self.alpha * observed + (1.0 - self.alpha) * previous


def route_execution(predicted_vram: int, vram_budget: int, predicted_time,
                    context_overhead: float, *,
                    vram_fraction: float = 0.5, time_multiple: float = 10.0) -> Execution:
    """Decide whether a GPU job runs fresh or in the warm worker.

    Fresh when its vram is at least ``vram_fraction`` of the device budget, or when its predicted
    time is at least ``time_multiple`` times the measured context-creation overhead. Warm
    otherwise. ``predicted_time`` of None (uncalibrated placement) holds the time trigger off, so
    the decision falls back to vram alone until the placement is calibrated.

    ``vram_fraction`` and ``time_multiple`` are tunables; ``context_overhead`` is the measured
    seconds to stand up a fresh CUDA context on the device.
    """
    if vram_budget <= 0:
        raise ValueError(f"vram_budget must be positive, got {vram_budget}")
    if predicted_vram >= vram_fraction * vram_budget:
        return Execution.FRESH
    if predicted_time is not None and predicted_time >= time_multiple * context_overhead:
        return Execution.FRESH
    return Execution.WARM
