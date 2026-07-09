# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Szabo's HFK calculator as a validator (SPEC 12.1 / 13.8; ADR 0006).

``knot_floer_homology`` (kfh) wraps Szabo's HFKcalc: given a planar diagram it computes HFK-hat,
the Ozsvath-Szabo tau invariant, and the Seifert genus. This adapter exposes it behind the one
validator contract, so it only ever cross-checks the native grid engine -- it never produces the
value a user receives (ADR 0006). The import is guarded: kfh is an optional cross-check, not a
runtime dependency.

Naming and conventions. kfh reports its own spellings (``tau``, ``seifert_genus``, ``ranks``);
``known_value`` returns them under the canonical invariant names the compute layer uses
(``ozsvath_szabo_tau``, ``three_genus``, ``knot_floer_homology``). kfh keys its HFK ranks
``(Alexander, Maslov)``; the canonical order (matching the native ``hfk_hat`` and KnotInfo's
``hfk_ranks``) is ``(Maslov, Alexander)``, so the ranks are transposed on the way out.
"""
from __future__ import annotations

from typing import Any

# Canonical invariant name -> kfh scalar field. HFK ranks are handled separately (they transpose).
_KFH_SCALAR = {
    "ozsvath_szabo_tau": "tau",
    "three_genus": "seifert_genus",
}
_COVERED = {"knot_floer_homology"} | set(_KFH_SCALAR)


class HFKValidator:
    """Szabo's HFK calculator (kfh) behind the validator contract (SPEC 12.1)."""

    name = "knot_floer_homology"
    covered_invariants = _COVERED

    def is_available(self) -> bool:
        try:
            import knot_floer_homology  # noqa: F401
        except ImportError:
            return False
        return True

    def version_info(self) -> dict:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _packageVersion

        try:
            return {"knot_floer_homology": _packageVersion("knot_floer_homology")}
        except PackageNotFoundError:
            return {"knot_floer_homology": "absent"}

    def known_value(self, knot, invariant: str) -> Any | None:
        """kfh's value for ``invariant`` under the canonical name/convention, or None.

        None means kfh cannot cross-check this input (invariant out of scope, or the knot carries no
        PD to run on) -- never a guessed value.
        """
        if invariant not in _COVERED or not knot.pd_code:
            return None
        out = raw_hfk(knot)
        if invariant == "knot_floer_homology":
            # kfh keys ranks (Alexander, Maslov); canonicalize to (Maslov, Alexander).
            return {
                (maslov, alexander): rank
                for (alexander, maslov), rank in out["ranks"].items()
            }
        return out[_KFH_SCALAR[invariant]]


def raw_hfk(knot) -> dict:
    """One kfh ``pd_to_hfk`` call on the knot's PD, returned raw (kfh's own field names,
    ranks keyed ``(Alexander, Maslov)``).

    The single place the library is invoked: the validator above and the comparison
    layer's measurement adapter (``scripts/comparison/adapters.kfhRun``) both delegate
    here, so the PD conversion and the call itself cannot drift apart.
    """
    import knot_floer_homology as kfh

    return kfh.pd_to_hfk([list(crossing) for crossing in knot.pd_code])
