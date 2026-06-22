# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Combinatorial knot Floer homology via grid diagrams (engine Phase 6).

A peer front end to Khovanov: a grid diagram emits graded F2 complexes (one per Alexander
grading) that the shared algebra back end reduces -- through the Phase 5 acceleration tiers
(bitint by default; parallel reduction across the independent gradings; GPU where present).
Provides the grid model, the Maslov / Alexander gradings, the empty-rectangle differentials
(bigraded and Alexander-filtered), the reduction to HFK-hat with the Seifert genus, the tau
invariant, parallel generation, and the synthetic staircase grid -- all validated
against KnotInfo, with the grid in the standard chirality so invariants match it directly.
"""
from .differential import differential, filtered_differential
from .gradings import alexander, alexander_euler_characteristic, maslov
from .grid import GridDiagram, staircase_grid
from .generation import grading_histogram, grid_complexes, parallel_grid_complexes
from .homology import grid_poincare, hfk_hat, reduce_complexes, seifert_genus
from .roster import floer_roster
from .tau import tau

__all__ = [
    "GridDiagram",
    "alexander",
    "alexander_euler_characteristic",
    "differential",
    "filtered_differential",
    "floer_roster",
    "grading_histogram",
    "grid_complexes",
    "grid_poincare",
    "hfk_hat",
    "maslov",
    "parallel_grid_complexes",
    "reduce_complexes",
    "seifert_genus",
    "staircase_grid",
    "tau",
]
