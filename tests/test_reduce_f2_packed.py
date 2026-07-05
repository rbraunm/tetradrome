# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

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
from tetradrome.algebra import gpu, reduce_f2_jit, tiers
from tetradrome.algebra.reduce_f2_jit import f2_rank_jit
from tetradrome.algebra.reduce_f2_packed import f2_rank_bitint, f2_rank_dense, f2_rank_words, pack_columns
from tetradrome.algebra.reduce_reference import f2_rank, homology
from tetradrome.engines import khovanov
from tetradrome.errors import BackendUnavailable

KNOTS = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_4"]

# Backends that can actually run in this environment (no GPU here, so not packed-gpu).
RUNNABLE = [name for name, ok, _ in tiers.available_f2_backends() if ok]


def test_handbuilt_ranks_match_reference():
    csc = pack_columns([{0, 1}, {1, 2}, {0, 2}])  # rank 2 over F2 (columns sum to zero)
    assert f2_rank_bitint(csc) == f2_rank(csc) == 2
    assert f2_rank_words(csc, 3, np) == 2


def test_empty_and_zero_columns():
    assert f2_rank_bitint(pack_columns([])) == f2_rank_words(pack_columns([]), 0, np) == 0
    zero = pack_columns([set(), set()])
    assert f2_rank_bitint(zero) == f2_rank_words(zero, 4, np) == 0


@pytest.mark.parametrize("name", KNOTS)
def test_packed_ranks_match_reference(name):
    pd = knots.from_name(name).pd_code
    for cx in khovanov.khovanov_complexes(pd).values():
        for n in cx.degrees():
            csc, nrows = cx.differential(n), cx.dim(n + 1)
            ref = f2_rank(csc)
            assert f2_rank_bitint(csc) == ref
            assert f2_rank_words(csc, nrows, np) == ref


@pytest.mark.parametrize("name", KNOTS)
def test_dense_and_jit_ranks_match_reference(name):
    # f2_rank_dense (the GPU kernel, here on numpy) and the jit reducer (here un-compiled)
    # must match the reference -- so the GPU and numba paths are validated by their shared
    # code even though neither device/compiler is present in this environment.
    pd = knots.from_name(name).pd_code
    for cx in khovanov.khovanov_complexes(pd).values():
        for n in cx.degrees():
            csc, nrows = cx.differential(n), cx.dim(n + 1)
            ref = f2_rank(csc)
            assert f2_rank_dense(csc, nrows, np) == ref
            assert f2_rank_jit(csc, nrows) == ref


def test_jit_tier_actually_compiles_when_numba_present():
    # The jit reducer silently runs the plain-Python impl when numba is absent, which keeps
    # the agreement tests above green even if the *compiled* tier is dead. That silent
    # fallback is only legitimate when numba is genuinely not installed. So: if numba IS
    # importable, the bound reducer MUST be a numba Dispatcher -- otherwise the acceleration
    # tier broke and degraded to interpreted Python while every other test still passed.
    # This is the guard that turns that invisible failure into a loud one.
    numba = pytest.importorskip("numba")
    from numba.core.dispatcher import Dispatcher

    f2_rank_jit(pack_columns([{0, 1}, {1, 2}, {0, 2}]), 3)  # force first-call binding/compilation
    assert reduce_f2_jit.HAVE_NUMBA, (
        "numba imports but reduce_f2_jit.HAVE_NUMBA is False -- presence check is broken; "
        "the jit tier will never compile."
    )
    assert isinstance(reduce_f2_jit._reducer, Dispatcher), (
        f"numba {numba.__version__} is installed but the jit reducer bound to "
        f"{type(reduce_f2_jit._reducer).__name__}, not a compiled Dispatcher -- the JIT "
        f"acceleration tier silently fell back to plain Python."
    )


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


# --- GPU dense-kernel agreement (only on a host with a usable CUDA device) -----------------
#
# The catalog knots above are far too small to be a meaningful GPU workload, and the packed
# reducers run the *same* code under numpy or cupy -- so small green tests don't prove the
# device actually did the work. These tests use large, deterministically-seeded F2 complexes
# big enough to be a real device job, assert the GPU kernel allocated device memory and ran
# on the configured card, and check it agrees with the pure-Python reference -- including a
# rank-DEFICIENT input, so the kernel is shown to find linear dependencies, not just report
# min(rows, cols).

_GPU = gpu.usable_cupy()
requires_gpu = pytest.mark.skipif(_GPU is None, reason="no usable CUDA device in this env")


def _cols_from_matrix(mat) -> list[set[int]]:
    """Column index-sets from a dense uint8 (nrows, ncols) F2 matrix."""
    return [set(np.nonzero(mat[:, j])[0].tolist()) for j in range(mat.shape[1])]


@requires_gpu
def test_gpu_dense_agrees_with_reference_on_large_full_rank_input():
    cp = _GPU
    n = 512
    rng = np.random.default_rng(0)                          # deterministic, reproducible
    mat = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
    csc = pack_columns(_cols_from_matrix(mat))
    ref = f2_rank(csc)                                      # pure-Python truth

    # Positive proof the work landed on the device. f2_rank_dense stores the matrix DENSE
    # as uint8 (nrows, ncols) -- not bit-packed (see reduce_f2_packed.f2_rank_dense:
    # host = np.zeros((nrows, ncols), uint8); mat = xp.asarray(host)). So the device array
    # is exactly nrows*ncols*itemsize bytes. We derive that floor from the real on-device
    # size of an identically-shaped array (and assert it IS a cupy device array), making the
    # byte bound a provable lower bound rather than a magic number that happens to pass --
    # the cupy pool only ever rounds UP, so >= floor_bytes cannot be met without a real
    # device allocation of the matrix.
    probe = cp.asarray(np.zeros((n, n), dtype=np.uint8))
    assert isinstance(probe, cp.ndarray)
    floor_bytes = probe.nbytes
    assert floor_bytes == n * n                             # dense uint8: 1 byte/element
    del probe

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    assert mempool.total_bytes() == 0
    rank = f2_rank_dense(csc, n, cp)
    cp.cuda.Stream.null.synchronize()
    assert mempool.total_bytes() >= floor_bytes, "GPU kernel did not allocate the device matrix"

    assert rank == ref


@requires_gpu
def test_gpu_dense_finds_dependencies_on_rank_deficient_input():
    cp = _GPU
    rng = np.random.default_rng(0)                          # deterministic, reproducible
    nrows, n_indep, n_dep = 256, 128, 64
    base = rng.integers(0, 2, size=(nrows, n_indep), dtype=np.uint8)
    # Each extra column is the XOR of two base columns, so it is linearly dependent on them:
    # the true rank cannot exceed n_indep, well below the n_indep + n_dep total columns.
    a = rng.integers(0, n_indep, size=n_dep)
    b = rng.integers(0, n_indep, size=n_dep)
    extra = base[:, a] ^ base[:, b]
    mat = np.concatenate([base, extra], axis=1)
    csc = pack_columns(_cols_from_matrix(mat))

    ref = f2_rank(csc)
    rank = f2_rank_dense(csc, nrows, cp)
    cp.cuda.Stream.null.synchronize()

    assert rank == ref
    assert rank <= n_indep < mat.shape[1]                   # deficiency really is present



def test_jit_without_numpy_fails_loud(monkeypatch):
    # numpy is an optional accel dep; with it absent the jit/packed reducer must fail loud
    # (BackendUnavailable), never a raw ImportError and never a silent degrade. Simulate the
    # dependency boundary by clearing the lazily-bound module global and its presence flag.
    monkeypatch.setattr(reduce_f2_jit, "np", None)
    monkeypatch.setattr(reduce_f2_jit, "HAVE_NUMPY", False)
    with pytest.raises(BackendUnavailable):
        f2_rank_jit(pack_columns([{0, 1}, {1, 2}]), 3)
