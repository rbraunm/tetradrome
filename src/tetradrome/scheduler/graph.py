# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The job DAG: a validated dependency graph that exposes which jobs are ready to run.

Dependencies are the only hard ordering in the scheduler; a job is ready once every job it
depends on has completed. The graph validates at construction -- unique keys, every dependency
refers to a real job, and no cycles -- so a malformed graph fails immediately rather than
deadlocking the scheduler later.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
import dataclasses

from .job import Job, Shard


def _producer_of(dep):
    """The producing job a dependency points at: a shard dependency points at its producer, a
    plain dependency is the job key itself. Topology (readiness, lineage, cycles) is in terms of
    producers; the shard distinction matters only to the executor, for which data to hand over."""
    return dep.producer if isinstance(dep, Shard) else dep


class JobGraph:
    """A validated DAG of jobs, keyed by ``Job.key``."""

    def __init__(self, jobs: Iterable[Job]):
        self._jobs: dict = {}
        for job in jobs:
            if job.key in self._jobs:
                raise ValueError(f"duplicate job key {job.key!r}")
            self._jobs[job.key] = job
        # The distinct producing jobs each job depends on -- shard dependencies collapsed to their
        # producer. This is the topology; job.dependencies (which may name shards) is the executor's
        # data spec. A shard dependency must point at a declared shard of a partitioned producer.
        self._producers: dict = {}
        for job in self._jobs.values():
            producers: set = set()
            for dep in job.dependencies:
                producer = _producer_of(dep)
                if producer not in self._jobs:
                    raise ValueError(f"job {job.key!r} depends on unknown job {producer!r}")
                if isinstance(dep, Shard):
                    target = self._jobs[producer]
                    if target.shards is None:
                        raise ValueError(
                            f"job {job.key!r} depends on shard {dep.key!r} of {producer!r}, "
                            f"which is not a partitioned job")
                    if dep.key not in target.shards:
                        raise ValueError(
                            f"job {job.key!r} depends on shard {dep.key!r} of {producer!r}, whose "
                            f"shards are {set(target.shards)}")
                producers.add(producer)
            self._producers[job.key] = frozenset(producers)
        self._dependents: dict = {key: [] for key in self._jobs}
        for key, producers in self._producers.items():
            for producer in producers:
                self._dependents[producer].append(key)
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        # Kahn's algorithm: if a topological order can't cover every node, there is a cycle.
        indegree = {key: len(self._producers[key]) for key in self._jobs}
        queue = deque(key for key, deg in indegree.items() if deg == 0)
        seen = 0
        while queue:
            key = queue.popleft()
            seen += 1
            for child in self._dependents[key]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if seen != len(self._jobs):
            raise ValueError("job graph has a dependency cycle")

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._jobs

    def get(self, key: Hashable) -> Job:
        return self._jobs[key]

    def jobs(self) -> tuple[Job, ...]:
        return tuple(self._jobs.values())

    def detach_inputs(self) -> dict:
        """Move every job's inputs out of the graph into a returned dict keyed by job, replacing
        each job with an input-stripped copy. The caller (the scheduler) becomes the sole owner of
        the inputs, so it can free each one as its job is dispatched and spill the heavy ones under
        pressure; the graph keeps only the cheap job structure. Single-use: after this the graph's
        jobs carry no inputs, so a graph is run once."""
        store = {}
        for key, job in self._jobs.items():
            store[key] = job.inputs
            self._jobs[key] = dataclasses.replace(job, inputs=None)
        return store

    def dependents(self, key: Hashable) -> tuple:
        """Keys of the jobs that depend directly on ``key``."""
        return tuple(self._dependents[key])

    def component(self, key: Hashable) -> frozenset:
        """Every job connected to ``key`` through dependency edges in either direction.

        For the per-knot DAG shape (a generation root, its per-grading reductions, the
        assembly) this is the whole knot: walking parents and children from any member reaches
        the root above and the siblings and assembly below. The scheduler uses it to abandon a
        knot's entire DAG when any of its jobs fails."""
        seen: set = set()
        stack = [key]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._producers[current])            # parents
            stack.extend(self._dependents[current])           # children
        return frozenset(seen)

    def ready(self, completed: Iterable[Hashable]) -> list[Job]:
        """Jobs whose dependencies are all completed and which are not themselves completed.
        The scheduler additionally filters out jobs already running."""
        done = set(completed)
        return [job for job in self._jobs.values()
                if job.key not in done and self._producers[job.key] <= done]
