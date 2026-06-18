# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""PD-code normalization and validation.

Fails loudly on malformed input -- we never repair or guess a PD code
(no silent fallbacks; decisions/0004).
"""
from __future__ import annotations

from .model import PDCode


def normalize(raw) -> PDCode:
    """Coerce a PD code to a tuple of 4-int tuples and validate its structure.

    A PD code for an n-crossing diagram has n entries, each a 4-tuple of arc
    labels, with every label appearing exactly twice across all entries, and the
    labels forming a contiguous run of 2n values (0-based, as Spherogram emits, or
    1-based). Anything else raises ValueError.
    """
    try:
        entries = [tuple(int(x) for x in entry) for entry in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PD code is not a sequence of integer tuples: {exc}") from exc

    if not entries:
        raise ValueError("PD code is empty.")

    for entry in entries:
        if len(entry) != 4:
            raise ValueError(f"PD entry {entry!r} does not have exactly 4 labels.")

    counts: dict[int, int] = {}
    for entry in entries:
        for label in entry:
            counts[label] = counts.get(label, 0) + 1

    offenders = {label: c for label, c in counts.items() if c != 2}
    if offenders:
        raise ValueError(
            f"PD labels must each appear exactly twice; offenders (label: count): {offenders}"
        )

    n = len(entries)
    distinct = sorted(counts)
    if distinct not in (list(range(0, 2 * n)), list(range(1, 2 * n + 1))):
        raise ValueError(
            f"PD labels are not a contiguous 0- or 1-based run of {2 * n} values: {distinct}"
        )

    return tuple(entries)  # type: ignore[return-value]
