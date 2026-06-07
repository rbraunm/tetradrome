"""Combinatorial knot Floer homology via grid diagrams (engine Phase 6).

A peer front end to Khovanov: a grid diagram emits graded F2 complexes (one per Alexander
grading) that the shared algebra back end reduces. Provides the grid model, the Maslov /
Alexander gradings, the empty-rectangle differential, and the reduction to HFK-hat with the
Seifert genus -- all validated against KnotInfo (the Alexander polynomial via the Euler
characteristic, HFK-hat up to mirror, and the three-genus). The tau invariant follows.
"""
from .differential import differential
from .gradings import alexander, alexander_euler_characteristic, maslov
from .grid import GridDiagram
from .homology import grid_poincare, hfk_hat, seifert_genus

__all__ = [
    "GridDiagram",
    "alexander",
    "alexander_euler_characteristic",
    "differential",
    "grid_poincare",
    "hfk_hat",
    "maslov",
    "seifert_genus",
]
