"""The shared, invariant-agnostic algebra back end (SPEC.md 13.6).

Front ends (Khovanov, Lee, later Floer) emit graded chain complexes; this package
reduces them to homology. F2 first (decision 0003); see roadmap/design/homology-engine.md.
"""
from .complex import GradedComplex
from .reduce_reference import f2_rank, homology

__all__ = ["GradedComplex", "f2_rank", "homology"]
