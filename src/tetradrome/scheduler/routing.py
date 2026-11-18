# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""What the scheduler has measured about its handlers, and the GPU warm-versus-fresh decision.

``Calibration`` holds the measurements: a work rate per job kind and device, and a dispatch
overhead per placement. Placement uses them to choose among a job's paths; the rest of this
module uses the GPU rate to route a GPU job to the warm worker or a fresh process.

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

from .job import Placement


class Execution(enum.Enum):
    """How a GPU job runs: in the warm held-context worker, or in a fresh process."""
    WARM = "warm"
    FRESH = "fresh"


def _device(placement) -> str:
    """Which code does the work. The inline path and both CPU placements run the same code, so
    they share a work rate; what separates them is dispatch overhead, which is measured apart."""
    return "gpu" if placement is Placement.GPU else "cpu"


class Calibration:
    """What is known about the handlers: benchmark figures, and an EWMA for anything unmeasured.

    The work rate is seconds per cost unit for a job kind on a device: ``observe`` folds a
    completed job's ``seconds / cost`` into it, and ``predicted_time(cost, placement, kind)`` is
    ``cost * rate`` once that kind has run on that device, else None. A job with no predicted cost
    (cost <= 0) carries no signal and is skipped.

    The dispatch overhead is the seconds a job spends being handed to a placement and reaped,
    beyond its own run time: zero for the inline path, measured for the others, None until then.

    ``fix`` sets the figures the benchmark measured (``tetradrome.calibrate``); those stay as
    given and a run's own timings no longer move them. A kind or placement the benchmark does not
    cover is learned while jobs run, for the life of the process. The engines share one
    calibration per process; a Scheduler given none starts a fresh one.
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._rate: dict = {}        # (kind, device) -> seconds per cost unit
        self._overhead: dict = {}    # placement -> dispatch seconds
        self._sibling: dict = {}     # kind -> time per unit sharing a core, over time alone
        self._fixed: set = set()     # (table name, key) of benchmark figures

    def _blend(self, name: str, table: dict, key, observed: float) -> None:
        if (name, key) in self._fixed:
            return
        previous = table.get(key)
        table[key] = (observed if previous is None
                      else self.alpha * observed + (1.0 - self.alpha) * previous)

    def fix(self, figures: dict) -> None:
        """Set the benchmark's ``rates``, ``overheads`` and ``sibling`` factors as final."""
        for entry in figures["sibling"]:
            self._sibling[entry["kind"]] = float(entry["factor"])
        for entry in figures["rates"]:
            key = (entry["kind"], entry["device"])
            self._rate[key] = float(entry["seconds_per_unit"])
            self._fixed.add(("rate", key))
        for entry in figures["overheads"]:
            placement = Placement[entry["placement"]]
            self._overhead[placement] = float(entry["seconds"])
            self._fixed.add(("overhead", placement))

    def rate(self, placement, kind: str = ""):
        """The current seconds-per-cost-unit estimate for a kind on a placement's device, or None
        if that kind has never run there."""
        return self._rate.get((kind, _device(placement)))

    def predicted_time(self, cost: float, placement, kind: str = ""):
        """Predicted run seconds for a job, or None if its kind has no rate on that device yet."""
        rate = self.rate(placement, kind)
        return None if rate is None else cost * rate

    def observe(self, cost: float, placement, seconds: float, kind: str = "") -> None:
        """Fold a completed job's measured run time into the rate for its kind and device."""
        if cost <= 0:
            return                   # no cost, so no per-unit signal to learn from
        self._blend("rate", self._rate, (kind, _device(placement)), seconds / cost)

    def shares(self, kind: str) -> bool:
        """Whether a job of ``kind`` should take the second thread of a busy physical core: only
        where the benchmark measured that two of them sharing a core do more than one alone, each
        running less than twice as slowly. Near that line the choice hardly matters, so noise in
        the measurement costs little whichever way it falls."""
        factor = self._sibling.get(kind)
        return factor is not None and factor < 2.0

    def overhead(self, placement):
        """Dispatch seconds for a placement: 0.0 inline, the measured figure otherwise, or None
        while it has not been measured."""
        if placement is Placement.INLINE:
            return 0.0
        return self._overhead.get(placement)

    def observe_overhead(self, placement, seconds: float) -> None:
        """Fold one job's measured dispatch-and-reap time into the placement's overhead."""
        self._blend("overhead", self._overhead, placement, seconds)


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
