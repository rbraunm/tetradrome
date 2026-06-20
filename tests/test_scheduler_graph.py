# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for the job value types and the DAG: validation, cycle detection, ready progression."""
import pytest

from tetradrome.scheduler import ComputePath, Job, JobGraph, Placement


def _cpu_path(cores=1, ram=10):
    return ComputePath(Placement.CPU_PINNED, cores=cores, ram_bytes=ram)


def _job(key, deps=()):
    return Job(key=key, run=lambda inputs, deps: None, inputs=None,
               paths=(_cpu_path(),), dependencies=deps)


# ---- ComputePath validation ----

def test_cpu_path_valid():
    p = ComputePath(Placement.CPU_PINNED, cores=2, ram_bytes=100)
    assert p.cores == 2 and p.vram_bytes == 0


def test_gpu_path_requires_vram():
    with pytest.raises(ValueError):
        ComputePath(Placement.GPU, cores=1, ram_bytes=100)  # vram defaults to 0


def test_gpu_path_valid():
    p = ComputePath(Placement.GPU, cores=1, ram_bytes=100, vram_bytes=8 << 30)
    assert p.vram_bytes == 8 << 30


def test_cpu_path_rejects_vram():
    with pytest.raises(ValueError):
        ComputePath(Placement.CPU_UNPINNED, cores=1, ram_bytes=100, vram_bytes=1)


def test_path_rejects_zero_cores():
    with pytest.raises(ValueError):
        ComputePath(Placement.CPU_PINNED, cores=0, ram_bytes=100)


# ---- Job validation/normalization ----

def test_job_requires_paths():
    with pytest.raises(ValueError):
        Job(key="x", run=lambda i, d: None, inputs=None, paths=())


def test_job_normalizes_paths_and_deps():
    j = Job(key="x", run=lambda i, d: None, inputs=None,
            paths=[_cpu_path()], dependencies=["a", "b"])
    assert isinstance(j.paths, tuple)
    assert j.dependencies == frozenset({"a", "b"})


# ---- JobGraph ----

def test_graph_rejects_duplicate_keys():
    with pytest.raises(ValueError):
        JobGraph([_job("a"), _job("a")])


def test_graph_rejects_dangling_dependency():
    with pytest.raises(ValueError):
        JobGraph([_job("b", deps=("a",))])


def test_graph_rejects_cycle():
    with pytest.raises(ValueError):
        JobGraph([_job("a", deps=("b",)), _job("b", deps=("a",))])


def test_graph_rejects_self_cycle():
    with pytest.raises(ValueError):
        JobGraph([_job("a", deps=("a",))])


def test_graph_dependents():
    g = JobGraph([_job("gen"), _job("rA", deps=("gen",)), _job("rB", deps=("gen",))])
    assert set(g.dependents("gen")) == {"rA", "rB"}
    assert g.dependents("rA") == ()


def test_ready_progression_floer_shape():
    # generation -> per-grading reductions -> assembly: the actual Floer DAG.
    g = JobGraph([
        _job("gen"),
        _job("rA", deps=("gen",)),
        _job("rB", deps=("gen",)),
        _job("asm", deps=("rA", "rB")),
    ])
    assert {j.key for j in g.ready(set())} == {"gen"}
    assert {j.key for j in g.ready({"gen"})} == {"rA", "rB"}
    assert {j.key for j in g.ready({"gen", "rA"})} == {"rB"}
    assert {j.key for j in g.ready({"gen", "rA", "rB"})} == {"asm"}
    assert {j.key for j in g.ready({"gen", "rA", "rB", "asm"})} == set()
