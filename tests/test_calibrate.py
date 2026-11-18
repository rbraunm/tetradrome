# SPDX-License-Identifier: Apache-2.0

"""The benchmark: it measures every kind of job compute() runs, once, and says when it does."""
import logging

import pytest

from tetradrome import calibrate
from tetradrome.scheduler import Calibration, Placement, calibration_store, detect_machine

# Every kind of job the engines build for compute(). A new kind belongs here and in its engine's
# benchmark(), or its first runs are placed without a figure.
_KINDS = [
    "f2.reduce.bitint",
    "floer.generation_merge",
    "floer.generation_slice",
    "floer.histogram_slice",
    "floer.tau_levels",
    "floer.tau_merge",
    "floer.tau_rank",
    "floer.tau_slice",
    "invariant.alexander_polynomial",
    "invariant.determinant",
    "invariant.jones_polynomial",
    "invariant.khovanov_homology",
    "invariant.rasmussen_s",
    "invariant.rational_khovanov_homology",
    "invariant.signature",
]


@pytest.fixture(scope="module")
def machine():
    return detect_machine()


@pytest.fixture(scope="module")
def first_use(machine, tmp_path_factory):
    """One first use against an empty store: (directory, calibration, figures, log records)."""
    directory = tmp_path_factory.mktemp("first-use")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    patch = pytest.MonkeyPatch()
    patch.setenv("TETRADROME_CACHE_DIR", str(directory))
    logging.getLogger("tetradrome.calibrate").addHandler(handler)
    try:
        calibration = Calibration()
        figures = calibrate.ensure(calibration, machine)
    finally:
        logging.getLogger("tetradrome.calibrate").removeHandler(handler)
        patch.undo()
    return directory, calibration, figures, records


def test_first_use_measures_every_kind_on_the_cpu(first_use):
    _, _, figures, _ = first_use
    assert [(entry["kind"], entry["device"]) for entry in figures["rates"]] == [
        (kind, "cpu") for kind in _KINDS]
    assert all(entry["seconds_per_unit"] > 0 for entry in figures["rates"])


def test_first_use_measures_each_worker_placement(first_use):
    _, _, figures, _ = first_use
    assert [entry["placement"] for entry in figures["overheads"]] == ["CPU_PINNED", "CPU_UNPINNED"]
    assert all(entry["seconds"] > 0 for entry in figures["overheads"])


def test_first_use_measures_sharing_a_core_for_every_kind_where_cores_have_two_threads(
        first_use, machine):
    _, _, figures, _ = first_use
    paired = any(len(core) >= 2 for node in machine.nodes for core in node.physical_cores)
    assert [entry["kind"] for entry in figures["sibling"]] == (_KINDS if paired else [])
    assert all(entry["factor"] > 0.5 for entry in figures["sibling"])


def test_first_use_says_it_is_measuring_and_why(first_use):
    _, _, _, records = first_use
    notices = [record.getMessage() for record in records if record.levelno == logging.WARNING]
    assert len(notices) == 2
    assert "this machine has not been measured yet" in notices[0]
    assert "python -m tetradrome.calibrate" in notices[0]


def test_first_use_keeps_its_figures_and_plans_with_them(first_use, machine):
    directory, calibration, figures, _ = first_use
    assert calibration_store.read(machine, directory) == (figures, "")
    rate = next(entry["seconds_per_unit"] for entry in figures["rates"]
                if entry["kind"] == "floer.generation_slice")
    assert calibration.predicted_time(1000.0, Placement.CPU_PINNED, "floer.generation_slice") == (
        1000.0 * rate)
    assert calibration.overhead(Placement.CPU_PINNED) == figures["overheads"][0]["seconds"]


def test_later_use_loads_without_measuring(first_use, machine, monkeypatch, caplog):
    directory, _, figures, _ = first_use
    monkeypatch.setenv("TETRADROME_CACHE_DIR", str(directory))
    with caplog.at_level(logging.INFO, logger="tetradrome.calibrate"):
        assert calibrate.ensure(Calibration(), machine) == figures
    assert caplog.records == []
