"""Combinatorial knot Floer homology via grid diagrams (engine Phase 6).

A peer front end to Khovanov: a grid diagram emits a graded chain complex that the shared
algebra back end reduces. This package currently provides the grid model and the Maslov /
Alexander gradings (validated against the Alexander polynomial via the graded Euler
characteristic); the rectangle-counting differential and HFK come next.
"""
from .grid import GridDiagram
from .gradings import alexander, alexander_euler_characteristic, maslov

__all__ = [
    "GridDiagram",
    "alexander",
    "alexander_euler_characteristic",
    "maslov",
]
