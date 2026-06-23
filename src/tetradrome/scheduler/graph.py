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

from .job import Job


class JobGraph:
    """A validated DAG of jobs, keyed by ``Job.key``."""

    def __init__(self, jobs: Iterable[Job]):
        self._jobs: dict = {}
        for job in jobs:
            if job.key in self._jobs:
                raise ValueError(f"duplicate job key {job.key!r}")
            self._jobs[job.key] = job
        self._dependents: dict = {key: [] for key in self._jobs}
        for job in self._jobs.values():
            for dep in job.dependencies:
                if dep not in self._jobs:
                    raise ValueError(f"job {job.key!r} depends on unknown job {dep!r}")
                self._dependents[dep].append(job.key)
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        # Kahn's algorithm: if a topological order can't cover every node, there is a cycle.
        indegree = {key: len(job.dependencies) for key, job in self._jobs.items()}
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
            stack.extend(self._jobs[current].dependencies)   # parents
            stack.extend(self._dependents[current])           # children
        return frozenset(seen)

    def ready(self, completed: Iterable[Hashable]) -> list[Job]:
        """Jobs whose dependencies are all completed and which are not themselves completed.
        The scheduler additionally filters out jobs already running."""
        done = set(completed)
        return [job for job in self._jobs.values()
                if job.key not in done and job.dependencies <= done]
