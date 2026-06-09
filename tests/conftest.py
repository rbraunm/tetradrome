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
own. See roadmap/design/floer-phase-6-plan.md and decisions/0011.
"""
from __future__ import annotations

import pytest

# --- the single heavy-tier gate -------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="run heavy tiers that need real compute (labradorite/local), not the "
        "ephemeral sandbox: @pytest.mark.heavy tests, and the wider validation rosters.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: needs real compute (labradorite/local); skipped unless --heavy is passed.",
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
