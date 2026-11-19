# SPDX-License-Identifier: Apache-2.0

"""Knot Floer homology HFK-hat over F2 by the bordered algebras of Ozsvath and Szabo
(arXiv:1603.06559): a type D structure for the top of a Morse presentation, a DA bimodule for
every slab below it, and the terminal module at the global minimum. It does not pay the grid's
n!; the grid engine is the reference it must agree with (roadmap/design/homology-engine.md,
section 7, Phase 10).
"""
from .knot import hfk_hat

__all__ = ["hfk_hat"]
