# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The validator registry: which computed oracles exist, which are wired, at what version.

This is the compute layer's single source of truth for the strict/soft/off doctrine
(ADR 0004): strict REQUIRES a computed oracle wherever one exists anywhere, and its
raise message must say whether the gap is provisioning ("not installed -- run
scripts/install_oracles.sh") or development ("not yet wired"). Answering that needs
three distinct states per invariant, and this module owns all three:

- exists anywhere:  ``computed_oracle_exists`` -- wired coverage OR an ``_UNWIRED`` entry.
- wired:            the validator instance is in ``_WIRED`` and covers the invariant.
- installed:        the wired validator's own ``is_available()`` probe.

Two sources of truth, deliberately split (derive, don't duplicate):

1. Wired coverage is LIVE. Everything about a wired oracle -- coverage, availability,
   version -- derives from its validator instance (``covered_invariants`` /
   ``is_available`` / ``version_info``, the SPEC 12.1 contract). Nothing is restated
   statically.
2. Unwired exists-anywhere is STATIC. ``_UNWIRED`` names computed oracles that exist in
   the world but that no validator wires yet -- exactly the set strict names in its
   "not yet wired" message. An entry is DELETED the day its oracle is wired; from then
   on the truth lives in that validator's ``covered_invariants``.

This registry is the validation path only. The comparison layer keeps its own oracle
list (``scripts/comparison/adapters.ORACLES``) for the benchmark artifact -- that is a
measurement harness, not a validator, and the relationship stays one-directional.
"""
from __future__ import annotations

from typing import Any, Protocol

from .hfk_adapter import HFKValidator
from .knotjob_adapter import KnotJobValidator
from .regina_adapter import ReginaValidator


class Validator(Protocol):
    """The SPEC 12.1 validator contract: read-only cross-checks, never producers.

    A validator never produces the value a user receives; it only confirms or
    contradicts the native one (ADR 0006).
    """

    name: str
    covered_invariants: set[str]

    def is_available(self) -> bool: ...
    def version_info(self) -> dict: ...
    def known_value(self, knot, invariant: str) -> Any | None: ...


# The validator instances actually wired into compute(). Order is consultation order.
_WIRED: tuple[Validator, ...] = (HFKValidator(), ReginaValidator(), KnotJobValidator())

# Computed oracles that exist in the world but are NOT yet wired as validators, per
# canonical invariant name (sourced from SPEC 12.3 / docs/backend_matrix.md and the
# provisioned set in scripts/install_oracles.sh). Floer has no entry: kfh is wired.
# pip SnapPy is Sage-only for every classical invariant here (verified empirically:
# Link.determinant/signature/alexander_polynomial/jones_polynomial all raise
# SageNotAvailable outside Sage), so it is subsumed by the sage entries rather than
# listed as a standalone oracle. Regina's Link API exposes no signature invariant.
_UNWIRED: dict[str, tuple[str, ...]] = {
    "determinant": ("sage",),
    "signature": ("sage",),
    "alexander_polynomial": ("sage",),
    "jones_polynomial": ("sage",),
    "khovanov_homology": ("javakh", "khoca"),
    "rational_khovanov_homology": ("javakh", "khoho", "khoca"),
    "rasmussen_s": ("khoca",),
}


def wired_validators(invariant: str) -> tuple[Validator, ...]:
    """The wired validators whose ``covered_invariants`` include ``invariant``."""
    return tuple(v for v in _WIRED if invariant in v.covered_invariants)


def unwired_oracles(invariant: str) -> tuple[str, ...]:
    """Computed oracles known to exist for ``invariant`` that no validator wires yet."""
    return _UNWIRED.get(invariant, ())


def computed_oracle_exists(invariant: str) -> bool:
    """Whether a computed oracle for ``invariant`` exists anywhere, wired or not.

    When True, strict REQUIRES one to run and pass; KnotInfo alone may validate a
    result only when this is False (fallback of last resort, ADR 0004).
    """
    return bool(wired_validators(invariant)) or bool(unwired_oracles(invariant))


def flat_version(validator: Validator) -> str:
    """``version_info()`` flattened to the single string a ValidatorRecord carries."""
    return ", ".join(
        f"{name} {version}" for name, version in sorted(validator.version_info().items())
    )
