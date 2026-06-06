"""Bridge to Spherogram for knot input and normalization.

Spherogram is the diagram layer: under plain pip it yields PD/DT codes without
Sage (research/backends-pip.md). It is a required dependency, so it is imported
directly; a missing install surfaces as a plain ImportError.
"""
from __future__ import annotations

import spherogram

from ..errors import UnknownKnot
from . import pd
from .model import NormalizedDiagram


def from_name(name: str) -> NormalizedDiagram:
    """Normalize a knot given by name (e.g. 'K11n34', '4_1').

    Raises UnknownKnot if Spherogram cannot resolve the name.
    """
    try:
        link = spherogram.Link(name)
    except Exception as exc:  # Spherogram raises a range of types for bad names
        raise UnknownKnot(f"Spherogram could not resolve knot name {name!r}: {exc}") from exc
    return NormalizedDiagram(
        pd_code=pd.normalize(link.PD_code()),
        source_notation="name",
        identity=name,
    )


def from_pd(pd_code, identity: str | None = None) -> NormalizedDiagram:
    """Normalize a knot given by raw PD code.

    Validates the PD code; raises ValueError if it is malformed.
    """
    return NormalizedDiagram(
        pd_code=pd.normalize(pd_code),
        source_notation="pd",
        identity=identity,
    )
