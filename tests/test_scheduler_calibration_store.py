# SPDX-License-Identifier: Apache-2.0

"""The scheduler's measurements kept between processes: stored once settled, reused while the
code is the same, measured again when it changes."""
import json

import pytest

from tetradrome.scheduler import Calibration, GPU, Machine, NumaNode, Placement
from tetradrome.scheduler import calibration_store

_GIB = 2**30


def _machine(cores=(0, 1), gpus=()):
    return Machine(nodes=(NumaNode(0, frozenset(cores), 8 * _GIB),), gpus=gpus,
                   mem_cap_bytes=8 * _GIB)


def _settled(seconds_per_unit=0.5, overhead=0.02, settle=1):
    calibration = Calibration(settle=settle)
    for _ in range(settle):
        calibration.observe(cost=2.0, placement=Placement.CPU_PINNED,
                            seconds=2.0 * seconds_per_unit, kind="slice")
        calibration.observe_overhead(Placement.CPU_PINNED, overhead)
    return calibration


def test_settled_figures_come_back_in_a_new_process(tmp_path):
    calibration_store.save(_settled(), _machine(), tmp_path, "code-a")
    fresh = Calibration()
    calibration_store.load(fresh, _machine(), tmp_path, "code-a")
    assert fresh.rate(Placement.CPU_PINNED, "slice") == 0.5
    assert fresh.rate(Placement.CPU_UNPINNED, "slice") == 0.5
    assert fresh.overhead(Placement.CPU_PINNED) == 0.02
    assert fresh.rate(Placement.GPU, "slice") is None


def test_figures_for_other_code_are_not_used(tmp_path):
    calibration_store.save(_settled(), _machine(), tmp_path, "code-a")
    fresh = Calibration()
    calibration_store.load(fresh, _machine(), tmp_path, "code-b")
    assert fresh.rate(Placement.CPU_PINNED, "slice") is None
    assert fresh.overhead(Placement.CPU_PINNED) is None


def test_new_code_replaces_the_old_figures(tmp_path):
    calibration_store.save(_settled(0.5), _machine(), tmp_path, "code-a")
    calibration_store.save(_settled(0.9), _machine(), tmp_path, "code-b")
    stored = json.loads(calibration_store.store_path(_machine(), tmp_path).read_text())
    assert stored["code"] == "code-b"
    assert [entry["seconds_per_unit"] for entry in stored["figures"]["rates"]] == [0.9]


def test_each_machine_has_its_own_file(tmp_path):
    small, large = _machine(cores=(0, 1)), _machine(cores=(0, 1, 2, 3))
    gpu = _machine(gpus=(GPU(0, 8 * _GIB),))
    paths = {calibration_store.store_path(machine, tmp_path) for machine in (small, large, gpu)}
    assert len(paths) == 3
    calibration_store.save(_settled(), small, tmp_path, "code-a")
    fresh = Calibration()
    calibration_store.load(fresh, large, tmp_path, "code-a")
    assert fresh.rate(Placement.CPU_PINNED, "slice") is None


def test_a_figure_is_stored_only_once_settled(tmp_path):
    calibration = Calibration(settle=3)
    for _ in range(2):
        calibration.observe(cost=1.0, placement=Placement.CPU_PINNED, seconds=1.0, kind="slice")
    calibration_store.save(calibration, _machine(), tmp_path, "code-a")
    assert not calibration_store.store_path(_machine(), tmp_path).exists()
    calibration.observe(cost=1.0, placement=Placement.CPU_PINNED, seconds=1.0, kind="slice")
    calibration_store.save(calibration, _machine(), tmp_path, "code-a")
    assert calibration_store.store_path(_machine(), tmp_path).exists()


def test_a_stored_figure_is_not_measured_again(tmp_path):
    calibration_store.save(_settled(0.5), _machine(), tmp_path, "code-a")
    fresh = Calibration(settle=1)
    calibration_store.load(fresh, _machine(), tmp_path, "code-a")
    fresh.observe(cost=1.0, placement=Placement.CPU_PINNED, seconds=7.0, kind="slice")
    assert fresh.rate(Placement.CPU_PINNED, "slice") == 0.5
    assert fresh.ready() == {"rates": [], "overheads": []}


def test_a_figure_another_process_stored_first_is_kept(tmp_path):
    calibration_store.save(_settled(0.5), _machine(), tmp_path, "code-a")
    calibration_store.save(_settled(0.9), _machine(), tmp_path, "code-a")
    fresh = Calibration()
    calibration_store.load(fresh, _machine(), tmp_path, "code-a")
    assert fresh.rate(Placement.CPU_PINNED, "slice") == 0.5


def test_figures_for_new_kinds_are_added_to_the_file(tmp_path):
    calibration_store.save(_settled(0.5), _machine(), tmp_path, "code-a")
    other = Calibration(settle=1)
    other.observe(cost=1.0, placement=Placement.CPU_PINNED, seconds=3.0, kind="reduce")
    calibration_store.save(other, _machine(), tmp_path, "code-a")
    fresh = Calibration()
    calibration_store.load(fresh, _machine(), tmp_path, "code-a")
    assert fresh.rate(Placement.CPU_PINNED, "slice") == 0.5
    assert fresh.rate(Placement.CPU_PINNED, "reduce") == 3.0


def test_an_unreadable_file_raises_naming_it(tmp_path):
    path = calibration_store.store_path(_machine(), tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match=path.name):
        calibration_store.load(Calibration(), _machine(), tmp_path, "code-a")


def test_store_directory_follows_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TETRADROME_CACHE_DIR", str(tmp_path))
    assert calibration_store.store_directory() == tmp_path
