"""The shared, invariant-agnostic algebra back end (SPEC.md 13.6).

Front ends (Khovanov, Lee, later Floer) emit graded chain complexes; this package
reduces them to homology. Two lanes behind one interface: an F2 fast lane (decision
0003) and a rational lane for what Lee / Rasmussen need. See
roadmap/design/homology-engine.md.
"""
from .complex import GradedComplex
from .gpu import detect_gpu, enablement_instructions, gpu_config
from .rational_complex import RationalComplex
from .rational_reduce import rational_homology, rational_rank
from .reduce_f2_packed import f2_rank_bitint, f2_rank_words
from .reduce_gaussian import gaussian_homology
from .reduce_reference import f2_rank, homology
from .tiers import available_f2_backends, best_available_backend, f2_homology, rank_backend

__all__ = [
    "GradedComplex",
    "RationalComplex",
    "available_f2_backends",
    "best_available_backend",
    "detect_gpu",
    "enablement_instructions",
    "f2_homology",
    "f2_rank",
    "f2_rank_bitint",
    "f2_rank_words",
    "gaussian_homology",
    "gpu_config",
    "homology",
    "rank_backend",
    "rational_homology",
    "rational_rank",
]
