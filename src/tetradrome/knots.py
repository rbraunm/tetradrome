# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Public knot input surface: `td.knots.from_name`, `from_pd`, `from_braid`.

A thin facade over the diagram layer (README usage; SPEC 13.10). `from_braid` is the
off-table path -- a braid word presents a knot whether or not it is tabulated.
"""
from .diagrams import NormalizedDiagram, from_braid, from_name, from_pd

__all__ = ["NormalizedDiagram", "from_braid", "from_name", "from_pd"]
