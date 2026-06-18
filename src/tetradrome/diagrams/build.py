# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Build a normalized diagram from a knot name or a raw PD code.

Tabulated knots are sourced from KnotInfo (its `pd_notation`); raw PD input is
validated directly and needs no backend. Spherogram is no longer required -- the
catalog source and the validation oracle are the same package (see
roadmap/research/knotinfo.md and the from_name discussion).
"""
from __future__ import annotations

from ..backends import knotinfo_backend
from . import pd
from .model import NormalizedDiagram


def from_name(name: str) -> NormalizedDiagram:
    """Normalize a tabulated knot given by name (e.g. 'K11n34', '4_1').

    The canonical identity is KnotInfo's spelling (conventions.md). Raises
    UnknownKnot if the name is not in KnotInfo; BackendUnavailable if KnotInfo is
    not installed.
    """
    identity = knotinfo_backend.normalize_name(name)
    pd_code = pd.normalize(knotinfo_backend.pd_notation(identity))
    return NormalizedDiagram(pd_code=pd_code, source_notation="name", identity=identity)


def from_pd(pd_code, identity: str | None = None) -> NormalizedDiagram:
    """Normalize a knot given by raw PD code. Validates it; raises ValueError if malformed."""
    return NormalizedDiagram(
        pd_code=pd.normalize(pd_code), source_notation="pd", identity=identity
    )


def from_braid(braid, identity: str | None = None) -> NormalizedDiagram:
    """Normalize a knot given by a braid word (e.g. [1, 1, 1] for the trefoil).

    `+j` means strand j crosses over j+1 (right-handed); `-j` means under. The closure
    of the braid is the knot. This is the off-table input path: a braid word presents
    a knot regardless of whether it is tabulated.

    `identity` is optional. Set it to a KnotInfo name when the braid presents a
    tabulated knot (it is canonicalized and enables oracle validation); leave it None
    for a knot KnotInfo does not contain, in which case computed invariants come back
    unvalidated (validate=True will raise; pass validate=False to opt in). Raises
    ValueError on a malformed braid word.
    """
    word = [int(v) for v in braid]
    if not word:
        raise ValueError("Braid word is empty.")
    if any(v == 0 for v in word):
        raise ValueError("Braid word entries must be nonzero.")
    canonical = (
        knotinfo_backend.normalize_name(identity) if identity is not None else None
    )
    return NormalizedDiagram(
        pd_code=(),
        source_notation="braid_word",
        identity=canonical,
        braid=tuple(word),
    )
