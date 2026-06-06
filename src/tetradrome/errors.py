"""Public error types.

No silent fallbacks: Tetradrome raises loudly rather than returning a plausible
wrong answer (decisions/0004-validate-by-default-error-policy). Additional members
of the error set (UnvalidatedResult, BackendUnavailable, ConventionMismatch,
ExportHashMismatch) are added here as the code that raises them lands.
"""


class TetradromeError(Exception):
    """Base class for all Tetradrome errors."""


class UnknownKnot(TetradromeError):
    """A knot identifier or diagram could not be resolved into a normalized diagram."""
