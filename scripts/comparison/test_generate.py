# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Tests for generate.py's per-oracle rendering logic.

These build synthetic Measurements and exercise the pure helpers -- capability detection, verdict
aggregation (mismatch dominates), the n/a and error cells, per-section column selection, and the
no-computing-oracle fallback -- so they run with no tetradrome, no oracle binaries, and no live
run. The actual measurement (measure / _tetraCell / _runOracles) is exercised by a real artifact
run on CT 250.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters   # noqa: E402
import generate   # noqa: E402

M = adapters.Measurement
LADDER = [("3_1", None), ("4_1", None)]        # 3_1 chiral, 4_1 amphichiral
LADDER_NONTORUS = [("4_1", None)]

# A Khovanov section as several oracles would report it: KnotJob/JavaKh/KhoHo/sage all do rational
# Khovanov (mirror on the chiral knot, pass on the amphichiral), KhoHo is n/a on the non-torus knot,
# and only KnotJob does s. sage additionally does Jones (a classical row).
ORACLE_RESULTS = {
    "knotjob": {
        "3_1": {"rational_khovanov_homology": M("total_rank=4", 0.30, agree="mirror"),
                "khovanov_homology": M("total_rank=6", None, agree="mirror"),
                "rasmussen_s": M("2", None, agree="mirror")},
        "4_1": {"rational_khovanov_homology": M("total_rank=6", 0.40, agree="pass"),
                "khovanov_homology": M("total_rank=10", None, agree="pass"),
                "rasmussen_s": M("0", None, agree="pass")},
    },
    "javakh": {
        "3_1": {"rational_khovanov_homology": M("total_rank=4", 0.10, agree="mirror")},
        "4_1": {"rational_khovanov_homology": M("total_rank=6", 0.12, agree="pass")},
    },
    "khoho": {
        "3_1": {"rational_khovanov_homology": M("total_rank=4", 0.50, agree="mirror")},
        "4_1": {"rational_khovanov_homology": M("n/a", None, note="non-torus", agree="n/a")},
    },
    "sage": {
        "3_1": {"rational_khovanov_homology": M("total_rank=4", None, agree="pass"),
                "khovanov_homology": M("total_rank=6", None, agree="pass"),
                "jones_polynomial": M("{...}", 3.0, agree="mirror")},
        "4_1": {"rational_khovanov_homology": M("total_rank=6", None, agree="pass"),
                "khovanov_homology": M("total_rank=10", None, agree="pass"),
                "jones_polynomial": M("{...}", 3.1, agree="mirror")},
    },
}


def test_computes_detects_capability_from_returned_keys():
    assert generate._computes(ORACLE_RESULTS["knotjob"], "rational_khovanov_homology")
    assert generate._computes(ORACLE_RESULTS["knotjob"], "rasmussen_s")
    assert not generate._computes(ORACLE_RESULTS["knotjob"], "jones_polynomial")
    assert not generate._computes(ORACLE_RESULTS["javakh"], "rasmussen_s")


def test_oracle_cell_verdict_mismatch_dominates_and_median_time():
    cell = generate._oracleCellFor(ORACLE_RESULTS["knotjob"], "rational_khovanov_homology", LADDER)
    assert cell["verdict"] == "mirror"                 # mirror (3_1) beats pass (4_1)
    assert cell["ms"] == 350.0                          # median(0.30, 0.40) s -> ms
    assert generate._oracleCellText(cell) == "350.00 ms \u2194"


def test_oracle_cell_all_pass_and_derived_no_time():
    cell = generate._oracleCellFor(ORACLE_RESULTS["sage"], "rational_khovanov_homology", LADDER)
    assert cell["verdict"] == "pass"
    assert cell["ms"] is None                           # sage's Khovanov rows are same-run
    assert generate._oracleCellText(cell) == "same call \u2713"


def test_oracle_cell_not_computed_is_dash():
    assert generate._oracleCellFor(ORACLE_RESULTS["javakh"], "rasmussen_s", LADDER) is None
    assert generate._oracleCellText(None) == "-"


def test_oracle_cell_na_when_applies_but_not_to_these_knots():
    cell = generate._oracleCellFor(
        ORACLE_RESULTS["khoho"], "rational_khovanov_homology", LADDER_NONTORUS)
    assert cell == {"na": True}
    assert generate._oracleCellText(cell) == "n/a"
    # but on a ladder including the torus knot it reports, not n/a
    mixed = generate._oracleCellFor(ORACLE_RESULTS["khoho"], "rational_khovanov_homology", LADDER)
    assert mixed.get("verdict") == "mirror" and mixed["ms"] == 500.0


def test_oracle_cell_mismatch_is_loud():
    perKnot = {"3_1": {"rasmussen_s": M("99", 0.2, agree="mismatch")}}
    cell = generate._oracleCellFor(perKnot, "rasmussen_s", [("3_1", None)])
    assert cell["verdict"] == "mismatch"
    assert "MISMATCH" in generate._oracleCellText(cell)


def test_oracle_cell_error_is_surfaced():
    perKnot = {"3_1": {"rasmussen_s": M("error: RuntimeError", None, agree="n/a")}}
    cell = generate._oracleCellFor(perKnot, "rasmussen_s", [("3_1", None)])
    assert cell == {"error": True}
    assert generate._oracleCellText(cell) == "\u2717 error"


def test_section_columns_are_all_computing_oracles_in_registry_order():
    khovInvs = generate.spec.invariants_for("knotjob")
    assert generate._sectionColumns(khovInvs, ORACLE_RESULTS) == [
        "knotjob", "javakh", "khoho", "sage"]


def test_fallback_column_for_sections_without_a_computing_oracle():
    oracleByKey = {orc.key: orc for orc in adapters.ORACLES}
    apex = types.SimpleNamespace(key="apex", oracle="(no peer)")
    assert generate._fallbackColumn(apex, {}, oracleByKey) == ("(no peer)", "no computing peer")
    khoca = types.SimpleNamespace(key="khoca", oracle="Khoca")
    assert generate._fallbackColumn(khoca, {"khoca": (True, "")}, oracleByKey) == (
        "Khoca", "adapter pending")
    assert generate._fallbackColumn(khoca, {"khoca": (False, "")}, oracleByKey) == (
        "Khoca", "absent (this run)")


if __name__ == "__main__":
    import traceback
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except Exception:
            failures += 1
            print("FAIL", test.__name__)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(tests) - failures, len(tests)))
    sys.exit(1 if failures else 0)
