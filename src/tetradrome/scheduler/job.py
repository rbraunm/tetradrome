# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Jobs and their compute paths: the unit of work the scheduler runs.

A job is a picklable callable plus its inputs, an ordered list of compute paths it supports
(fastest first), and the keys of the jobs it depends on. A path declares where it can run and
what it costs there: a GPU path (device VRAM plus the host core and RAM that feed it), a
node-pinned CPU path (cores and RAM on one NUMA node), or an unpinned CPU path (cores and RAM
that may span nodes, slower). The scheduler runs the fastest path the machine can serve; order
in the list is the speed ranking.

Jobs run in separate processes, so the callable must be importable (top-level, not a closure)
and its inputs and result must pickle. A job is invoked as ``run(inputs, deps)`` where ``deps``
maps each dependency key to that job's result.
"""
from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Hashable


class Placement(enum.Enum):
    """Where a compute path runs."""
    GPU = "gpu"                    # a CUDA device, plus host cores/RAM to feed it
    CPU_PINNED = "cpu_pinned"      # cores + RAM bound to a single NUMA node
    CPU_UNPINNED = "cpu_unpinned"  # cores + RAM, may span nodes (no locality, slower)


@dataclasses.dataclass(frozen=True)
class ComputePath:
    """One way a job can run: where, and what it costs there.

    ``cores`` and ``ram_bytes`` are the host footprint (for a GPU path, the cores and RAM that
    feed the device). ``vram_bytes`` is device memory and is meaningful only for a GPU path.
    """
    placement: Placement
    cores: int
    ram_bytes: int
    vram_bytes: int = 0

    def __post_init__(self):
        if self.cores < 1:
            raise ValueError(f"a compute path needs at least one core (got {self.cores})")
        if self.ram_bytes < 0:
            raise ValueError(f"ram_bytes must be non-negative (got {self.ram_bytes})")
        if self.vram_bytes < 0:
            raise ValueError(f"vram_bytes must be non-negative (got {self.vram_bytes})")
        if self.placement is Placement.GPU and self.vram_bytes <= 0:
            raise ValueError("a GPU path must declare positive vram_bytes")
        if self.placement is not Placement.GPU and self.vram_bytes != 0:
            raise ValueError("only a GPU path may declare vram_bytes")


@dataclasses.dataclass(frozen=True)
class Job:
    """A unit of work: identity, the callable + inputs, its compute paths, its dependencies.

    ``paths`` is ordered fastest to slowest and must be non-empty. ``run`` is invoked as
    ``run(inputs, deps)`` and returns the job's result, where ``deps`` maps each dependency
    key to its result. ``key`` is the job's identity and the handle its result is stored under.
    """
    key: Hashable
    run: Callable
    inputs: object
    paths: tuple[ComputePath, ...]
    dependencies: frozenset = dataclasses.field(default_factory=frozenset)

    def __post_init__(self):
        # Accept any iterable for paths/dependencies; store normalized immutables.
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "dependencies", frozenset(self.dependencies))
        if not self.paths:
            raise ValueError(f"job {self.key!r} declares no compute paths")
        if not callable(self.run):
            raise ValueError(f"job {self.key!r} run is not callable")
