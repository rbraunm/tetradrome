"""Runtime tier registry for F2 reduction (engine Phase 5).

The pure-Python reference is the always-present floor; faster tiers are optional and
detected at runtime, so the library runs anywhere and accelerates where it can (design
section 3). Every tier must return the *identical* answer as the reference -- that is the
correctness discipline, exercised by tests and by the benchmark's accuracy pass.

Backends, fastest-capable last:
- ``reference``    set-based reducer (the floor), always available
- ``bitint``       pure-Python int bit-vectors, always available, faster than reference
- ``packed-cpu``   numpy uint64 word arrays, when numpy is importable
- ``packed-gpu``   cupy uint64 word arrays on a CUDA GPU, when cupy + a device are present

``packed-cpu`` and ``packed-gpu`` run the *same* `f2_rank_words` code with a different
array module, so the CPU run validates the exact path the GPU takes.
"""
from __future__ import annotations

from .reduce_f2_packed import f2_rank_bitint, f2_rank_words
from .reduce_reference import f2_rank

BACKENDS = ("reference", "bitint", "packed-cpu", "packed-gpu")


def _numpy():
    try:
        import numpy

        return numpy
    except ImportError:
        return None


def _cupy_if_gpu():
    """cupy module iff it imports and reports at least one CUDA device, else None."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        return None
    return None


def available_f2_backends() -> list[tuple[str, bool, str]]:
    """Ordered ``(name, available, note)`` for every known backend."""
    np = _numpy()
    cp = _cupy_if_gpu()
    return [
        ("reference", True, "pure-Python set reducer (the floor)"),
        ("bitint", True, "pure-Python int bit-vectors"),
        ("packed-cpu", np is not None,
         "numpy uint64 word arrays" if np is not None else "needs numpy"),
        ("packed-gpu", cp is not None,
         "cupy uint64 word arrays on GPU" if cp is not None else "needs cupy + CUDA GPU"),
    ]


def rank_backend(name: str):
    """Return a ``(columns, nrows) -> rank`` callable for `name`, or raise if the
    backend is unavailable. `nrows` is ignored by backends that do not need it."""
    if name == "reference":
        return lambda columns, nrows=0: f2_rank(columns)
    if name == "bitint":
        return lambda columns, nrows=0: f2_rank_bitint(columns)
    if name == "packed-cpu":
        np = _numpy()
        if np is None:
            raise RuntimeError("packed-cpu backend needs numpy (pip install numpy).")
        return lambda columns, nrows=0: f2_rank_words(columns, nrows, np)
    if name == "packed-gpu":
        cp = _cupy_if_gpu()
        if cp is None:
            raise RuntimeError("packed-gpu backend needs cupy and a CUDA GPU.")
        return lambda columns, nrows=0: f2_rank_words(columns, nrows, cp)
    raise ValueError(f"unknown F2 backend {name!r}; choose from {BACKENDS}.")


def best_available_backend() -> str:
    """Fastest-capable backend present: packed-gpu > packed-cpu > bitint."""
    avail = {name: ok for name, ok, _ in available_f2_backends()}
    for name in ("packed-gpu", "packed-cpu", "bitint"):
        if avail.get(name):
            return name
    return "reference"


def f2_homology(cx, backend: str = "bitint") -> dict[int, int]:
    """F2 homology of a GradedComplex via the chosen rank backend. Identical result to
    `reduce_reference.homology` for every backend. Pass ``backend="auto"`` for the
    fastest available. Does not re-verify d^2 = 0 -- the complex owns that."""
    if backend == "auto":
        backend = best_available_backend()
    rank_fn = rank_backend(backend)
    ranks = {n: rank_fn(cx.differential(n), cx.dim(n + 1)) for n in cx.degrees()}
    result: dict[int, int] = {}
    for n in cx.degrees():
        h = cx.dim(n) - ranks.get(n, 0) - ranks.get(n - 1, 0)
        if h < 0:
            raise RuntimeError(
                f"negative Betti number at degree {n}: input is not a valid complex."
            )
        if h:
            result[n] = h
    return result
