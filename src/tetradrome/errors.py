# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Public error types.

No silent fallbacks: Tetradrome raises loudly rather than returning a plausible
wrong answer (decisions/0004-validate-by-default-error-policy). The remaining
members of the error set (ConventionMismatch, ExportHashMismatch) are added here as
the code that raises them lands.
"""


class TetradromeError(Exception):
    """Base class for all Tetradrome errors."""


class UnknownKnot(TetradromeError):
    """A knot identifier or diagram could not be resolved into a normalized diagram."""


class BackendUnavailable(TetradromeError):
    """A required backend or data source is not installed/available."""


class UnvalidatedResult(TetradromeError):
    """A result could not be validated under validate="strict" or "soft" (decisions/0004)."""
