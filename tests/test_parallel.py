"""Tests for multi-core reduction (engine Phase 5).

Parallelism must not change answers, only timing -- so the contract is parallel == serial.
`workers=2` forces the process-pool path even on a single-CPU host, so the agreement is
exercised rather than skipped via the in-process shortcut.
"""
import pytest

from tetradrome import knots
from tetradrome.algebra import tiers
from tetradrome.algebra.parallel import parallel_f2_homology
from tetradrome.engines import khovanov

KNOTS = ["3_1", "4_1", "5_2", "6_2", "7_4"]


def _batch():
    items = {}
    for name in KNOTS:
        pd = knots.from_name(name).pd_code
        for j, cx in khovanov.khovanov_complexes(pd).items():
            items[(name, j)] = cx
    return items


def _serial(items, backend):
    return {key: tiers.f2_homology(cx, backend=backend) for key, cx in items.items()}


@pytest.mark.parametrize("backend", ["reference", "bitint", "packed-cpu"])
def test_parallel_matches_serial(backend):
    items = _batch()
    got = parallel_f2_homology(items, backend=backend, workers=2)  # force the pool
    assert got == _serial(items, backend)


def test_single_worker_path_matches():
    items = _batch()
    assert parallel_f2_homology(items, backend="bitint", workers=1) == _serial(items, "bitint")


def test_iterable_input_accepted():
    items = _batch()
    pairs = list(items.items())
    assert parallel_f2_homology(pairs, backend="bitint", workers=2) == _serial(items, "bitint")


def test_single_item_skips_pool():
    pd = knots.from_name("3_1").pd_code
    j, cx = next(iter(khovanov.khovanov_complexes(pd).items()))
    assert parallel_f2_homology({j: cx}, backend="bitint", workers=4) == {
        j: tiers.f2_homology(cx, backend="bitint")
    }


def test_gpu_backend_rejected():
    with pytest.raises(ValueError):
        parallel_f2_homology({}, backend="packed-gpu", workers=2)
