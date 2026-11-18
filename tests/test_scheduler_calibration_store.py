# SPDX-License-Identifier: Apache-2.0

"""The store of benchmark figures: used while the machine and the software are the ones measured,
and says why not otherwise."""
import json

import pytest

from tetradrome.scheduler import GPU, Calibration, Machine, NumaNode, Placement, calibration_store

_GIB = 2**30
_FIGURES = {
    "rates": [{"kind": "slice", "device": "cpu", "seconds_per_unit": 0.5}],
    "overheads": [{"placement": "CPU_PINNED", "seconds": 0.02}],
    "sibling": [{"kind": "slice", "factor": 1.5}],
    "worker": {"ram_baseline": 123, "context_vram": 0},
}


def _machine(cores=(0, 1), ram=8 * _GIB, gpus=()):
    return Machine(nodes=(NumaNode(0, frozenset(cores), ram),), gpus=gpus, mem_cap_bytes=ram)


def test_written_figures_are_read_back(tmp_path):
    calibration_store.write(_machine(), tmp_path, _FIGURES)
    assert calibration_store.read(_machine(), tmp_path) == (_FIGURES, "")


def test_an_empty_store_says_the_machine_is_unmeasured(tmp_path):
    assert calibration_store.read(_machine(), tmp_path) == (
        None, "this machine has not been measured yet")


def test_other_processors_or_gpus_are_another_machine(tmp_path):
    calibration_store.write(_machine(cores=(0, 1)), tmp_path, _FIGURES)
    for other in (_machine(cores=(0, 1, 2, 3)), _machine(gpus=(GPU(0, 8 * _GIB),))):
        assert calibration_store.read(other, tmp_path) == (
            None, "this machine's processors or GPUs are not the ones measured before")


def test_ram_does_not_make_another_machine(tmp_path):
    calibration_store.write(_machine(ram=8 * _GIB), tmp_path, _FIGURES)
    assert calibration_store.read(_machine(ram=64 * _GIB), tmp_path) == (_FIGURES, "")


@pytest.mark.parametrize("change", [
    {"source": "0" * 64},
    {"python": "0.0.0"},
    {"libraries": {"numpy": "0.0.0"}},
])
def test_changed_software_makes_the_figures_stale(tmp_path, change):
    path = calibration_store.write(_machine(), tmp_path, _FIGURES)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(change) < set(stored["software"])
    stored["software"].update(change)
    path.write_text(json.dumps(stored), encoding="utf-8")
    assert calibration_store.read(_machine(), tmp_path) == (
        None, "the code or its libraries changed since the last measurement")


def test_writing_replaces_the_earlier_figures(tmp_path):
    calibration_store.write(_machine(), tmp_path, _FIGURES)
    later = {**_FIGURES, "rates": [{"kind": "slice", "device": "cpu", "seconds_per_unit": 0.9}]}
    calibration_store.write(_machine(), tmp_path, later)
    assert calibration_store.read(_machine(), tmp_path) == (later, "")
    assert len(list(tmp_path.iterdir())) == 1


def test_an_unreadable_file_raises_naming_it(tmp_path):
    path = calibration_store.store_path(_machine(), tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match=path.name):
        calibration_store.read(_machine(), tmp_path)


def test_store_directory_follows_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TETRADROME_CACHE_DIR", str(tmp_path))
    assert calibration_store.store_directory() == tmp_path


def test_fixed_figures_are_not_moved_by_what_a_run_measures():
    calibration = Calibration()
    calibration.fix(_FIGURES)
    calibration.observe(cost=1.0, placement=Placement.CPU_PINNED, seconds=7.0, kind="slice")
    calibration.observe_overhead(Placement.CPU_PINNED, 3.0)
    assert calibration.rate(Placement.CPU_PINNED, "slice") == 0.5
    assert calibration.rate(Placement.CPU_UNPINNED, "slice") == 0.5
    assert calibration.overhead(Placement.CPU_PINNED) == 0.02
    assert calibration.rate(Placement.GPU, "slice") is None


def test_a_kind_the_benchmark_does_not_cover_is_still_learned():
    calibration = Calibration()
    calibration.fix(_FIGURES)
    calibration.observe(cost=2.0, placement=Placement.CPU_PINNED, seconds=3.0, kind="other")
    assert calibration.rate(Placement.CPU_PINNED, "other") == 1.5


def test_hyperthread_layout_is_part_of_the_machine(tmp_path):
    cpus = frozenset({0, 1})
    paired = Machine(nodes=(NumaNode(0, cpus, _GIB, (cpus,)),), gpus=(), mem_cap_bytes=_GIB)
    calibration_store.write(_machine(ram=_GIB), tmp_path, _FIGURES)
    assert calibration_store.read(paired, tmp_path) == (
        None, "this machine's processors or GPUs are not the ones measured before")


@pytest.mark.parametrize("factor, expected", [(1.0, True), (1.5, True), (1.81, True),
                                               (1.83, False), (2.0, False), (3.0, False)])
def test_second_thread_is_used_only_where_two_jobs_do_a_tenth_more_than_one(factor, expected):
    calibration = Calibration()
    calibration.fix({**_FIGURES, "sibling": [{"kind": "slice", "factor": factor}]})
    assert calibration.shares("slice") is expected


def test_a_kind_with_no_shared_core_figure_does_not_share():
    calibration = Calibration()
    calibration.fix(_FIGURES)
    assert calibration.shares("other") is False
