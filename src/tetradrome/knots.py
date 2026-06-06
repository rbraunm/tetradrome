"""Public knot input surface: `td.knots.from_name` / `td.knots.from_pd`.

A thin facade over the diagram layer (README usage; SPEC 13.10).
"""
from .diagrams import NormalizedDiagram, from_name, from_pd

__all__ = ["NormalizedDiagram", "from_name", "from_pd"]
