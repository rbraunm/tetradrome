# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""The shared, invariant-agnostic algebra back end (SPEC.md 13.6).

Front ends (Khovanov, Lee, later Floer) emit graded chain complexes; this package
reduces them to homology. Two lanes behind one interface: an F2 fast lane (decision
0003) and a rational lane for what Lee / Rasmussen need. See
roadmap/design/homology-engine.md.
"""
from .complex import GradedComplex
from .gpu import detect_gpu, enablement_instructions, gpu_config
from .memory import ComplexSize, Routing, dense_block_bytes, dense_block_ops, dense_reduction_bytes, grading_cost_ops, grading_peak_bytes, max_grading_bytes, predict_cost, predict_size, route_backend
from .multimodular import rational_homology_multimodular, rational_rank_multimodular
from .rational_complex import RationalComplex
from .rational_reduce import rational_homology, rational_rank
from .reduce_f2_jit import f2_rank_jit
from .reduce_f2_packed import f2_rank_bitint, f2_rank_dense, f2_rank_words
from .reduce_gaussian import gaussian_homology
from .reduce_reference import f2_kernel, f2_rank, homology
from .tiers import available_f2_backends, best_available_backend, f2_homology, rank_backend

__all__ = [
    "GradedComplex",
    "RationalComplex",
    "available_f2_backends",
    "best_available_backend",
    "ComplexSize",
    "Routing",
    "dense_block_bytes",
    "dense_block_ops",
    "dense_reduction_bytes",
    "detect_gpu",
    "enablement_instructions",
    "f2_homology",
    "f2_kernel",
    "f2_rank",
    "f2_rank_bitint",
    "f2_rank_dense",
    "f2_rank_jit",
    "f2_rank_words",
    "gaussian_homology",
    "gpu_config",
    "grading_cost_ops",
    "grading_peak_bytes",
    "homology",
    "max_grading_bytes",
    "predict_cost",
    "predict_size",
    "rank_backend",
    "rational_homology",
    "rational_homology_multimodular",
    "rational_rank",
    "rational_rank_multimodular",
    "route_backend",
]
