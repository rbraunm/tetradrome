"""Tests for the bit-packed F2 reducers and the tier registry (engine Phase 5).

The whole point of an acceleration tier is that it is faster *and indistinguishable* from
the reference. So every test here is an agreement test: bit-packed ranks must equal the
set-based reference rank, and tier-dispatched homology must equal the reference homology,
across the real Khovanov catalog. Speed is measured by the benchmark script, not asserted
here.
"""
import numpy as np
import pytest

from tetradrome import knots
from tetradrome.algebra import tiers
from tetradrome.algebra.reduce_f2_packed import f2_rank_bitint, f2_rank_words
from tetradrome.algebra.reduce_reference import f2_rank, homology
from tetradrome.engines import khovanov

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]

# Backends that can actually run in this environment (no GPU here, so not packed-gpu).
RUNNABLE = [name for name, ok, _ in tiers.available_f2_backends() if ok]


def test_handbuilt_ranks_match_reference():
    cols = [{0, 1}, {1, 2}, {0, 2}]  # rank 2 over F2 (columns sum to zero)
    assert f2_rank_bitint(cols) == f2_rank(cols) == 2
    assert f2_rank_words(cols, 3, np) == 2


def test_empty_and_zero_columns():
    assert f2_rank_bitint([]) == f2_rank_words([], 0, np) == 0
    assert f2_rank_bitint([set(), set()]) == f2_rank_words([set(), set()], 4, np) == 0


@pytest.mark.parametrize("name", KNOTS)
def test_packed_ranks_match_reference(name):
    pd = knots.from_name(name).pd_code
    for cx in khovanov.khovanov_complexes(pd).values():
        for n in cx.degrees():
            cols, nrows = cx.differential(n), cx.dim(n + 1)
            ref = f2_rank(cols)
            assert f2_rank_bitint(cols) == ref
            assert f2_rank_words(cols, nrows, np) == ref


@pytest.mark.parametrize("name", KNOTS)
@pytest.mark.parametrize("backend", RUNNABLE)
def test_tier_homology_matches_reference(name, backend):
    pd = knots.from_name(name).pd_code
    for cx in khovanov.khovanov_complexes(pd).values():
        assert tiers.f2_homology(cx, backend=backend) == homology(cx, verify=False)


def test_registry_reports_floor_backends():
    avail = dict((name, ok) for name, ok, _ in tiers.available_f2_backends())
    assert avail["reference"] and avail["bitint"]          # always present
    assert avail["packed-cpu"]                             # numpy is installed here


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        tiers.rank_backend("packed-quantum")


def test_unavailable_gpu_backend_raises_cleanly():
    # No GPU in this environment, so requesting it must fail loudly, not silently degrade.
    if not dict((n, ok) for n, ok, _ in tiers.available_f2_backends())["packed-gpu"]:
        with pytest.raises(RuntimeError):
            tiers.rank_backend("packed-gpu")


def test_auto_selects_an_available_backend():
    assert tiers.best_available_backend() in RUNNABLE
