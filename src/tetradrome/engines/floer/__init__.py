"""Combinatorial knot Floer homology via grid diagrams (engine Phase 6).

A peer front end to Khovanov: a grid diagram emits graded F2 complexes (one per Alexander
grading) that the shared algebra back end reduces. Provides the grid model, the Maslov /
Alexander gradings, the empty-rectangle differentials (bigraded and Alexander-filtered), the
reduction to HFK-hat with the Seifert genus, and the tau invariant -- all validated against
KnotInfo (the Alexander polynomial via the Euler characteristic, HFK-hat, the three-genus,
and tau). The grid is taken in the standard chirality, so the invariants match KnotInfo
directly.
"""
from .differential import differential, filtered_differential
from .gradings import alexander, alexander_euler_characteristic, maslov
from .grid import GridDiagram
from .homology import grid_poincare, hfk_hat, seifert_genus
from .tau import tau

__all__ = [
    "GridDiagram",
    "alexander",
    "alexander_euler_characteristic",
    "differential",
    "filtered_differential",
    "grid_poincare",
    "hfk_hat",
    "maslov",
    "seifert_genus",
    "tau",
]
