"""The shared, invariant-agnostic algebra back end (SPEC.md 13.6).

Front ends (Khovanov, Lee, later Floer) emit graded chain complexes; this package
reduces them to homology. Two lanes behind one interface: an F2 fast lane (decision
0003) and a rational lane for what Lee / Rasmussen need. See
roadmap/design/homology-engine.md.
"""
from .complex import GradedComplex
from .rational_complex import RationalComplex
from .rational_reduce import rational_homology, rational_rank
from .reduce_reference import f2_rank, homology

__all__ = [
    "GradedComplex",
    "RationalComplex",
    "f2_rank",
    "homology",
    "rational_homology",
    "rational_rank",
]
