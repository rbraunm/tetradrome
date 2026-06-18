# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tetradrome: a reproducible, audit-friendly workbench for knot invariants.

See SPEC.md for the full design. Public surface grows as components land; today it
exposes the knot input layer (`tetradrome.knots`).
"""
from ._version import __version__
from . import invariants, knots

__all__ = ["invariants", "knots", "__version__"]
