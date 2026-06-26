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
class Shard:
    """A reference to one shard of a partitioned job's output. A job whose ``shards`` is set
    returns ``{shard_key: payload}`` and the scheduler stores each shard as an independent held
    result; a consumer that needs only one shard depends on ``Shard(producer_key, shard_key)``
    instead of the whole producer. For scheduling -- readiness and lineage -- a shard dependency
    is a dependency on the producing job; for data and memory it addresses just the one shard, so
    a wide consumer never pulls or re-materializes the producer's whole output."""
    producer: Hashable
    key: Hashable


@dataclasses.dataclass(frozen=True)
class Job:
    """A unit of work: identity, the callable + inputs, its compute paths, its dependencies.

    ``paths`` is ordered fastest to slowest and must be non-empty. ``run`` is invoked as
    ``run(inputs, deps)`` and returns the job's result, where ``deps`` maps each dependency
    key to its result. ``key`` is the job's identity and the handle its result is stored under.
    ``cost`` is the predicted work in abstract units (the builder sets it, e.g. from
    ``predict_cost``); the scheduler compares it to measured runtime to calibrate and to decide
    whether a job is substantial enough to warrant its own process. Zero means unpredicted.

    ``output_bytes`` is the declared size of the job's result, the portion of its working set that
    persists in the parent after the job exits and is held until the last consumer drains it. It is
    charged against global RAM like a working-set footprint -- ``max(declared, measured)``, with an
    over-budget warning when the measured result exceeds the declaration -- so a too-low estimate
    is a tuning signal, not a silent under-charge. Zero means unpredicted: the measured size is
    charged reactively.

    ``shards``, when set, declares the job's output is partitioned: ``run`` returns
    ``{shard_key: payload}`` for exactly these keys, and the scheduler stores each shard as an
    independent held result a consumer addresses via ``Shard(key, shard_key)``. None means a single
    whole result. ``output_bytes`` for a partitioned job is the declared total across shards.

    ``device_resident`` declares the job runs on the GPU and returns a device buffer to keep
    resident in VRAM rather than copy back to the host: the result stays in the warm worker's CUDA
    context and a GPU consumer reads it on-device, avoiding the round trip. It requires every path
    to be a GPU path (the producer always runs on the device) and is incompatible with ``shards``
    (a device-resident result is whole). It is an optimization, not a guarantee: when routing
    defeats it -- the producer routes fresh, or a consumer is not GPU-only -- the scheduler warns
    and falls back (the result becomes host-resident, or a fresh GPU consumer is run warm to read
    the buffer) rather than failing.
    """
    key: Hashable
    run: Callable
    inputs: object
    paths: tuple[ComputePath, ...]
    dependencies: frozenset = dataclasses.field(default_factory=frozenset)
    cost: float = 0.0
    output_bytes: int = 0
    shards: frozenset | None = None
    device_resident: bool = False
    device_resident: bool = False

    def __post_init__(self):
        # Accept any iterable for paths/dependencies; store normalized immutables.
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "dependencies", frozenset(self.dependencies))
        if self.shards is not None:
            object.__setattr__(self, "shards", frozenset(self.shards))
            if not self.shards:
                raise ValueError(f"job {self.key!r} declares partitioned output with no shards")
        if not self.paths:
            raise ValueError(f"job {self.key!r} declares no compute paths")
        if not callable(self.run):
            raise ValueError(f"job {self.key!r} run is not callable")
        if self.device_resident:
            if self.shards is not None:
                raise ValueError(f"job {self.key!r} cannot be both device-resident and partitioned")
            if any(path.placement is not Placement.GPU for path in self.paths):
                raise ValueError(
                    f"job {self.key!r} is device-resident but has a non-GPU path; a device-resident "
                    f"result is produced on the device, so every path must be a GPU path")
