"""Tetradrome: a reproducible, audit-friendly workbench for knot invariants.

See SPEC.md for the full design. Public surface grows as components land; today it
exposes the knot input layer (`tetradrome.knots`).
"""
from . import knots

__all__ = ["knots"]
__version__ = "0.0.0"
