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
