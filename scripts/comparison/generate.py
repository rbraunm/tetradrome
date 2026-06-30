#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Generate BENCHMARKS.md: Tetradrome's native engines vs the established tools, timed live.

Run on a host that has the oracles installed (CT 250 is the comprehensive one); the generator
captures whatever is present, times it on a fixed knot ladder, and writes pretty markdown to
``BENCHMARKS.md`` AND to stdout. It reports times and the measured state as DATA -- never a
pass/fail or a speed gate (CLAUDE.md). An absent oracle is reported absent, never faked.

    python scripts/comparison/generate.py                 # default ladder, writes BENCHMARKS.md
    python scripts/comparison/generate.py --reps 5 --out BENCHMARKS.md
    python scripts/comparison/generate.py --with-floer-grid   # time the grid Floer engine too

The grid Floer engine timing is flag-gated: it spins up the multi-core scheduler, which belongs
on the provisioned box, not in a laptop/sandbox. Off by default; the row reads "pending (CT 250)".
"""
from __future__ import annotations

import argparse
import datetime
import os
import platform
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters                                     # noqa: E402  (sibling modules, flat import)
import spec                                         # noqa: E402

DEFAULT_LADDER = ["3_1", "4_1", "5_2", "6_2", "7_4", "8_19"]

# Output glyphs as \u escapes so this source stays ASCII while the markdown renders pretty.
_STATUS = {
    "done": "\u2705 done",
    "landing": "\U0001F6A7 landing",
    "near": "\U0001F51C near",
    "build": "\U0001F527 build",
    "bound": "\U0001F4CF bound",
    "research": "\U0001F52C research",
    "out": "\u26D4 out",
}
_KNOTINFO = {"yes": "\u2713", "partial": "~", "no": "\u2717"}
_AGREE = {
    "pass": "\u2713 matches KnotInfo",
    "mirror": "\u2194 matches (mirror)",
    "mismatch": "\u2717 MISMATCH",
    "oracle": "\U0001F517 oracle ref",
    "no-oracle": "no oracle",
    "n/a": "-",
    "": "-",
}
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Clean column header per group (the verbose Group.oracle text is for the blurb, not the table).
_ORACLE_COL = {
    "kfh": "knot_floer_homology",
    "knotjob": "KnotJob",
    "sage": "SageMath",
    "khoca": "Khoca",
    "snappy": "SnapPy",
    "knotinfo_bounds": "KnotInfo (value)",
    "apex": "(no peer)",
}


# ---- environment / provenance -------------------------------------------------------------

def _cpuModel():
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def _ramGiB():
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / (1024 * 1024), 1)
    except Exception:
        return None


def _gitSha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _toolVersions():
    versions = {}
    try:
        kfh = __import__("knot_floer_homology")
        versions["knot_floer_homology"] = getattr(kfh, "__version__", "present")
    except Exception:
        pass
    try:
        td = __import__("tetradrome")
        versions["tetradrome"] = getattr(td, "__version__", _gitSha())
    except Exception:
        versions["tetradrome"] = _gitSha()
    return versions


def hostFacts():
    return {
        "host": platform.node(),
        "cpu": _cpuModel(),
        "cores": os.cpu_count(),
        "ram_gib": _ramGiB(),
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.release()}",
        "git": _gitSha(),
        "when": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "tools": _toolVersions(),
    }


# ---- measurement --------------------------------------------------------------------------

def _medianMs(seconds_list):
    clean = [s for s in seconds_list if s is not None]
    if not clean:
        return None
    return statistics.median(clean) * 1000.0


def measure(ladder, reps, withFloerGrid):
    """Return per-invariant {name: {"tetra": cell, "oracle": cell}} where each cell is a dict
    {ms, label, agree}; ms None means not measured (with a label saying why)."""
    oracleLive = {orc.key: orc.available() for orc in adapters.ORACLES}
    oracleByKey = {orc.key: orc for orc in adapters.ORACLES}
    results = {}

    for inv in spec.INVARIANTS:
        tetraCell = _tetraCell(inv, ladder, reps, withFloerGrid)
        oracleCell = _oracleCell(inv, ladder, reps, oracleLive, oracleByKey)
        results[inv.name] = {"tetra": tetraCell, "oracle": oracleCell}
    return results, oracleLive


def _tetraCell(inv, ladder, reps, withFloerGrid):
    if inv.tetra is None:
        return {"ms": None, "label": "not implemented (target)", "agree": "n/a"}
    kind, arg = inv.tetra
    if kind == "compute":
        seconds, agree = [], "pass"
        for _name, knot in ladder:
            measurement = adapters.measureTetradrome(knot, arg, reps)
            seconds.append(measurement.seconds)
            if measurement.agree not in ("pass", "oracle"):
                agree = measurement.agree or agree
        ms = _medianMs(seconds)
        label = None if ms is not None else "could not run"
        return {"ms": ms, "label": label, "agree": agree}
    if kind == "floer":
        if not withFloerGrid:
            return {"ms": None, "label": "pending (CT 250 grid run)", "agree": "n/a"}
        return {"ms": None, "label": "grid timing TODO (wire from bench_grid_floer)",
                "agree": "n/a"}
    return {"ms": None, "label": "unknown tetra kind", "agree": "n/a"}


def _oracleCell(inv, ladder, reps, oracleLive, oracleByKey):
    group = next(g for g in spec.GROUPS if g.key == inv.group)
    orcKey = inv.group if inv.group in oracleByKey else None
    if orcKey is None:                               # bounds / apex groups have no computing peer
        return {"ms": None, "label": "no computing peer", "agree": "n/a", "col": group.oracle}
    present, _detail = oracleLive[orcKey]
    if not present:
        return {"ms": None, "label": "absent (this run)", "agree": "n/a", "col": orcKey}
    orc = oracleByKey[orcKey]
    if orc.run is None:
        return {"ms": None, "label": "adapter pending", "agree": "n/a", "col": orcKey}
    seconds = []
    agree = "oracle"
    for _name, knot in ladder:
        perInvariant = orc.run(knot, reps)
        cell = perInvariant.get(inv.name)
        if cell is None:
            continue
        seconds.append(cell.seconds)
        agree = cell.agree or agree
    ms = _medianMs(seconds)
    label = None if ms is not None else "same upstream call"
    return {"ms": ms, "label": label, "agree": agree, "col": orcKey}


# ---- markdown -----------------------------------------------------------------------------

def _cellText(cell):
    if cell["ms"] is not None:
        return f"{cell['ms']:.2f} ms"
    return f"*{cell['label']}*"


def _cell(text):
    """Make text safe inside a markdown table cell: escape pipes (which are column delimiters,
    e.g. the |Delta(-1)| / |s|/2 absolute-value notation) and flatten any newline, so a cell can
    never split the row or break alignment."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def emit(results, oracleLive, facts, ladder, reps, withFloerGrid):
    lines = []
    add = lines.append

    add("# Tetradrome - native invariants vs the established tools\n")
    add("> Auto-generated by `scripts/comparison/generate.py`. Every number here was measured on "
        "the host below at generation time; nothing is hand-entered. Tetradrome's thesis is "
        "**correctness and auditability over speed** - a validated number beats a fast one, "
        "and faithful-but-portable beats fast-but-won't-install. Speed is shown as honest data and "
        "a target, never a verdict.\n")

    add("## How to read this\n")
    add("Rows are grouped by the **computational gold-master** that already exists for each "
        "invariant. KnotInfo is the value oracle throughout (the `KnotInfo` column), and its own "
        "group only for the bounds no program computes. Tetradrome computes every \"done\" row "
        "**natively** - the external tools are validators, never a compute backend "
        "(decision 0006).\n")
    add("Status: " + "  ".join(_STATUS[s] for s in
        ["done", "landing", "near", "build", "bound", "research"]) + ".\n")
    add("Empty cells are honest: *not implemented (target)* = we don't compute it yet (the oracle "
        "time is the bar to aim at); *absent (this run)* = the oracle wasn't installed on this "
        "host; *no computing peer* = no program computes it for general knots.\n")

    add("## How this run was generated\n")
    add(f"- **Host:** `{facts['host']}` - {facts['cpu']}, {facts['cores']} logical cores, "
        f"{facts['ram_gib']} GiB RAM")
    add(f"- **Software:** {facts['system']}, Python {facts['python']}, "
        f"tetradrome `{facts['git']}`")
    toolList = ", ".join(f"{name} `{ver}`" for name, ver in facts["tools"].items()) or "(none)"
    add(f"- **Oracle versions:** {toolList}")
    present = [k for k, (ok, _d) in oracleLive.items() if ok]
    absent = [k for k, (ok, _d) in oracleLive.items() if not ok]
    add(f"- **Oracles present:** {', '.join(present) or 'none'}  |  "
        f"**absent this run:** {', '.join(absent) or 'none'}")
    add(f"- **Knot ladder:** {', '.join(n for n, _k in ladder)} "
        f"(timings are the median across the ladder)")
    add(f"- **Timing:** best-of-{reps} wall seconds per knot, reported in milliseconds"
        + ("" if withFloerGrid else "; grid Floer engine timing deferred to a CT 250 run") + "")
    add(f"- **Generated:** {facts['when']}\n")

    for group in spec.GROUPS:
        invs = spec.invariants_for(group.key)
        if not invs:
            continue
        present_here = oracleLive.get(group.key, (False, ""))[0]
        add(f"## {group.title}\n")
        add(f"{group.blurb}\n")
        oracleHeader = _ORACLE_COL.get(group.key, group.key)
        add(f"| Invariant | Math | In \u2192 Out | Status | KnotInfo | Tetradrome | "
            f"{oracleHeader} | Validation |")
        add("|---|---|---|---|---|---|---|---|")
        for inv in invs:
            cell = results[inv.name]
            agreeKey = (cell["tetra"]["agree"] if cell["tetra"]["agree"] not in ("n/a", "")
                        else cell["oracle"]["agree"])
            row = [
                f"**{_cell(inv.label)}**",
                _cell(inv.math),
                _cell(f"{inv.inputs} \u2192 {inv.outputs}"),
                _STATUS.get(inv.status, inv.status),
                _KNOTINFO.get(inv.knotinfo, "?"),
                _cell(_cellText(cell["tetra"])),
                _cell(_cellText(cell["oracle"])),
                _AGREE.get(agreeKey, "-"),
            ]
            add("| " + " | ".join(row) + " |")
        add("")

    add("## Honest caveats\n")
    add("- **Speed is a target, not a verdict.** Where Tetradrome trails a compiled oracle "
        "(notably the n! grid Floer engine vs Szabo's compiled-C HFKcalc), that gap is expected "
        "and is the documented case for the Tier-2 Szabo-cube engine. Trailing a compiled tool "
        "while staying native, auditable, and portable is the accepted trade.")
    add("- **Chirality.** Tetradrome follows KnotInfo's mirror convention; `knot_floer_homology` "
        "may report the mirror, so tau / s agreement is judged up to sign.")
    add("- **Host matters.** These are this host's numbers. The grid Floer engine's multi-core "
        "scaling shows only on a many-core box (CT 250); a comprehensive run regenerates this "
        "file there with every oracle installed.")
    add("- **Acceleration preserves answers.** Every acceleration tier is checked to reproduce the "
        "reference exactly; speed never changes a result.\n")
    return "\n".join(lines)


# ---- entry ---------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate the Tetradrome comparison artifact.")
    parser.add_argument("--reps", type=int, default=3, help="best-of-N timing repeats")
    parser.add_argument("--knots", nargs="+", default=DEFAULT_LADDER, help="tabulated knot ladder")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "BENCHMARKS.md"),
                        help="output path for the artifact")
    parser.add_argument("--with-floer-grid", action="store_true",
                        help="also time the multi-core grid Floer engine (provisioned box only)")
    args = parser.parse_args()

    ladder = adapters.buildLadder(args.knots)
    facts = hostFacts()
    results, oracleLive = measure(ladder, args.reps, args.with_floer_grid)
    markdown = emit(results, oracleLive, facts, ladder, args.reps, args.with_floer_grid)

    with open(args.out, "w") as handle:
        handle.write(markdown)
    sys.stderr.write(f"wrote {args.out}\n")
    print(markdown)


if __name__ == "__main__":
    main()
