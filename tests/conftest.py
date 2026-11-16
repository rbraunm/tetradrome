# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Shared pytest fixtures and the heavy-tier gate.

Default runs must be safe in an ephemeral, memory- and time-limited sandbox. Some
validation tiers -- large grid-Floer rosters today, large-input cases on other engines
later -- need real compute (labradorite or a capable local box) and must not run by
default. They hang off a single ``--heavy`` flag shared across all engines:

  * Tests marked ``@pytest.mark.heavy`` are skipped unless ``--heavy`` is given.
  * Roster-parametrized agreement tests widen their tier under ``--heavy`` (the grid Floer
    roster goes from n <= 8 by default to n <= 10 under the flag). Each engine maps the one
    flag to its own (default, heavy) thresholds via ``pytest_generate_tests`` below.

So ``pytest`` is the sandbox-safe suite and ``pytest --heavy`` (on labradorite) is the full
sweep. The gate is environmental -- "is this safe in the cheap ephemeral env?" -- not
per-engine, so a new engine with a scaling wall reuses the same flag rather than adding its
own. See roadmap/design/homology-engine.md section 7.
"""
from __future__ import annotations

import os

import pytest



@pytest.fixture(scope="session", autouse=True)
def _scheduler_calibration_store(tmp_path_factory):
    """Keep the suite's scheduler measurements out of the user's cache and start it measuring:
    tests run the code being changed, whose figures are not the installed code's."""
    previous = os.environ.get("TETRADROME_CACHE_DIR")
    os.environ["TETRADROME_CACHE_DIR"] = str(tmp_path_factory.mktemp("calibration"))
    yield
    if previous is None:
        del os.environ["TETRADROME_CACHE_DIR"]
    else:
        os.environ["TETRADROME_CACHE_DIR"] = previous


# --- the single heavy-tier gate -------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="run heavy tiers that need real compute (labradorite/local), not the "
        "ephemeral sandbox: @pytest.mark.heavy tests, and the wider validation rosters.",
    )


# Minimum cores for the heavy tier to be worth running -- below this it would crawl serially,
# so refuse rather than pretend. NUMA pinning and CUDA are optional accelerators the engine
# uses automatically when present; the hard requirement here is real CPU parallelism.
_MIN_HEAVY_CORES = 4


def _core_count() -> int:
    try:
        return len(os.sched_getaffinity(0))   # respects container cpusets
    except AttributeError:                     # not Linux
        return os.cpu_count() or 1


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: needs real compute (labradorite/local); skipped unless --heavy is passed.",
    )
    if _heavy(config) and _core_count() < _MIN_HEAVY_CORES:
        raise pytest.UsageError(
            f"--heavy needs real parallel compute (>= {_MIN_HEAVY_CORES} cores); this "
            f"environment has {_core_count()}. Run it on labradorite or a capable local box."
        )


def _heavy(config) -> bool:
    return bool(config.getoption("--heavy"))


def pytest_collection_modifyitems(config, items):
    if _heavy(config):
        return
    skip = pytest.mark.skip(reason="heavy tier; pass --heavy to run (labradorite/local)")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip)


# --- shared validation rosters (derived from KnotInfo, never hardcoded) ---------------
# floer_roster lives in the package (tetradrome.engines.floer) so the sweep script shares it;
# imported lazily in pytest_generate_tests to keep collection independent of the optional
# KnotInfo backend.

# Floer tiers: sandbox-safe default vs the full acceptance sweep.
_FLOER_MAX_N = {"default": 8, "heavy": 10}


def pytest_generate_tests(metafunc):
    """Parametrize any test that requests ``floer_knot`` over the derived roster: n <= 8 by
    default, n <= 10 under --heavy. If KnotInfo is absent the floer tests skip (an optional
    backend, like the GPU/pinning skips) rather than breaking collection for the suite."""
    if "floer_knot" not in metafunc.fixturenames:
        return
    from tetradrome.engines.floer import floer_roster
    from tetradrome.errors import BackendUnavailable

    max_n = _FLOER_MAX_N["heavy" if _heavy(metafunc.config) else "default"]
    try:
        roster = floer_roster(max_n)
    except BackendUnavailable:
        metafunc.parametrize(
            "floer_knot",
            [pytest.param(None, marks=pytest.mark.skip(reason="KnotInfo backend unavailable"))],
        )
        return
    metafunc.parametrize(
        "floer_knot", roster, ids=[f"{name}_n{n}" for name, n in roster]
    )


# --- the roster computed together -----------------------------------------------------
# The roster tests check one knot each, but the knots are computed as one batch through the
# scheduler, so they share the machine instead of each waiting for the one before it. A fixture
# computes only the knots the selected tests ask it about.

def _selected_grids(request, fixture_name: str) -> dict:
    from tetradrome.engines.floer.grid import GridDiagram

    names = []
    for item in request.session.items:
        if fixture_name not in getattr(item, "fixturenames", ()):
            continue
        knot = getattr(item, "callspec", None) and item.callspec.params.get("floer_knot")
        if knot and knot[0] not in names:
            names.append(knot[0])
    return {name: GridDiagram.from_knotinfo(name) for name in names}


@pytest.fixture(scope="session")
def floer_hfk(request):
    """``(ranks by knot, failure text by knot)`` for the roster knots under test."""
    from tetradrome.engines.floer import hfk_hat_many

    return hfk_hat_many(_selected_grids(request, "floer_hfk"))


@pytest.fixture(scope="session")
def floer_tau(request):
    """``(tau by knot, failure text by knot)`` for the roster knots under test."""
    from tetradrome.engines.floer import tau_many

    return tau_many(_selected_grids(request, "floer_tau"))
