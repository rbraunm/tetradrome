#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm
"""Regenerate BENCHMARKS.md end to end.

Runs the comparison generator on the fully provisioned box (CT 250) via tools/ct_exec.py,
pulls the artifact back, writes it locally, then commits and pushes it. The execution path
and extraction are encoded here once so they are not re-derived by hand each time.

This runs on the workstation that can reach CT 250 (Toshiro), not in a sandbox. The numbers
in the artifact are DATA, never a gate (see CLAUDE.md); this tool just produces and commits
the chart.

Order is fail-fast: git push-readiness is checked and the local branch is synced with origin
(fast-forward, or rebase if it diverged) BEFORE the multi-minute CT 250 run -- so a misconfigured
or out-of-date git never wastes a benchmark and the final push fast-forwards. Every seam fails
loud with a clear next step; nothing is swallowed and there is no silent fallback.

    python tools/generate_benchmarks.py
    python tools/generate_benchmarks.py --reps 5 --with-floer-grid
    python tools/generate_benchmarks.py --ref claude --no-push

The CT 250 path knobs (--ctid, --ref, --src, --python) default to the known box layout.
Override the relevant flag rather than editing this file if the box changes.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from shutil import which
from typing import NoReturn

# Windows consoles default to a non-UTF-8 codepage (e.g. cp1252) that cannot encode the artifact's
# status emoji. Make our own console UTF-8 so printing diagnostics never crashes; the artifact
# itself is transported base64-encoded (see build_remote_command) so it never reaches a console as
# raw text in the first place.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CT_EXEC = os.path.join(HERE, "ct_exec.py")

BEGIN = "<<<TETRA_BENCHMARKS_BEGIN>>>"
END = "<<<TETRA_BENCHMARKS_END>>>"
REMOTE_LOG = "/tmp/tetra_bm_gen.log"
DEFAULT_BRANCHES = {"main", "master"}


def die(message: str) -> NoReturn:
    sys.stderr.write("ERROR: " + message + "\n")
    raise SystemExit(1)


def note(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


# --- local git ------------------------------------------------------------------------------

def git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command in the repo root, capturing text output. Never raises on a non-zero
    exit; callers inspect returncode so the failure message can be specific. Decoded UTF-8 with
    replacement so a stray byte in git output never crashes the capture on a non-UTF-8 console."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          encoding="utf-8", errors="replace")


def preflight_git() -> str:
    """Verify git is usable and push-ready before any CT 250 work. Returns the branch name."""
    if which("git") is None:
        die("git is not on PATH. Install git or fix PATH, then re-run.")

    inside = git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        die("not inside a git work tree (looked in %s). Run from the tetradrome repo." % REPO_ROOT)

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if not branch or branch == "HEAD":
        die("detached HEAD or no current branch. Check out a working branch (e.g. claude) and re-run.")
    if branch in DEFAULT_BRANCHES:
        die("on the default branch '%s'. Refusing to commit benchmarks there; check out a "
            "working branch (e.g. claude) and re-run." % branch)

    name = git("config", "user.name").stdout.strip()
    email = git("config", "user.email").stdout.strip()
    if not name or not email:
        die("git identity not configured. Set user.name and user.email "
            "(git config user.name ... / git config user.email ...) and re-run.")

    ls = git("ls-remote", "--heads", "origin")
    if ls.returncode != 0:
        die("cannot reach 'origin' (auth or network). Configure the git remote/credentials "
            "and re-run.\n" + ls.stderr.strip())

    return branch


def remote_branch_exists(branch: str) -> bool:
    ls = git("ls-remote", "--heads", "origin", branch)
    if ls.returncode != 0:
        die("cannot query origin for branch '%s':\n%s" % (branch, ls.stderr.strip()))
    return bool(ls.stdout.strip())


def _ahead_behind(branch: str) -> tuple[int, int]:
    """(ahead, behind) of local HEAD relative to origin/<branch>. Assumes origin/<branch> is
    up to date locally (fetch first)."""
    ahead = git("rev-list", "--count", "origin/%s..HEAD" % branch).stdout.strip() or "0"
    behind = git("rev-list", "--count", "HEAD..origin/%s" % branch).stdout.strip() or "0"
    return int(ahead), int(behind)


def sync_with_remote(branch: str) -> None:
    """Bring the local branch in line with origin/<branch> BEFORE the run, so the eventual push
    fast-forwards. Handles the three cases the artifact push actually hits:

      - origin/<branch> does not exist yet -> nothing to sync; the push will create it.
      - local is simply behind             -> fast-forward.
      - local has diverged (unpushed work) -> rebase onto origin, aborting cleanly on conflict.

    Fails loud on a dirty tree (fast-forward refused) or a rebase conflict rather than guessing."""
    if not remote_branch_exists(branch):
        note("origin/%s does not exist yet; it will be created on push." % branch)
        return
    fetched = git("fetch", "origin", branch)
    if fetched.returncode != 0:
        die("git fetch origin %s failed:\n%s" % (branch, fetched.stderr.strip()))
    ahead, behind = _ahead_behind(branch)
    if behind == 0:
        note("local '%s' is current with origin (ahead %d)." % (branch, ahead))
        return
    if ahead == 0:
        note("origin/%s is ahead by %d commit(s); fast-forwarding local." % (branch, behind))
        ff = git("merge", "--ff-only", "origin/%s" % branch)
        if ff.returncode != 0:
            die("fast-forward failed (uncommitted local changes in the way?). Commit or stash "
                "them and re-run.\n%s" % (ff.stderr + "\n" + ff.stdout).strip())
        return
    note("local '%s' has diverged from origin (ahead %d, behind %d); rebasing onto origin."
         % (branch, ahead, behind))
    rebase = git("rebase", "origin/%s" % branch)
    if rebase.returncode != 0:
        git("rebase", "--abort")
        die("rebase onto origin/%s hit a conflict and was aborted -- your branch is unchanged. "
            "Likely an unpushed local commit touches the same file as origin. Resolve it manually "
            "(or drop the unpushed commit) and re-run.\n%s"
            % (branch, (rebase.stdout + "\n" + rebase.stderr).strip()))
    note("rebased local '%s' onto origin." % branch)


# --- remote run -----------------------------------------------------------------------------

def build_remote_command(src: str, python: str, ref: str, reps: int, with_floer: bool) -> str:
    """One remote shell line. Each step gates the next with &&. The generator's console output
    goes to a log; the artifact is returned base64-encoded between sentinels so non-ASCII content
    (the status emoji) survives transport through any console codepage."""
    src = src.rstrip("/")
    out = src + "/BENCHMARKS.md"
    floer = " --with-floer-grid" if with_floer else ""
    return (
        "cd {src} && "
        "git fetch --depth 1 origin {ref} && "
        "git reset --hard FETCH_HEAD && "
        "{python} scripts/comparison/generate.py --reps {reps}{floer} --out {out} "
        "> /dev/null 2> {log} && "
        "printf '{begin}\\n' && base64 -w0 {out} && printf '\\n{end}\\n'"
    ).format(src=src, ref=ref, python=python, reps=reps, floer=floer,
             out=out, log=REMOTE_LOG, begin=BEGIN, end=END)


def ct_exec(ctid: int, remote_command: str, timeout: int) -> subprocess.CompletedProcess:
    if not os.path.exists(CT_EXEC):
        die("tools/ct_exec.py not found beside this script (%s)." % CT_EXEC)
    cmd = [sys.executable, CT_EXEC, "--ctid", str(ctid), "--", remote_command]
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        die("CT %d run exceeded %ds. Raise --remote-timeout or check the box." % (ctid, timeout))


def fetch_remote_log(ctid: int) -> str:
    result = ct_exec(ctid, "base64 -w0 " + REMOTE_LOG, timeout=60)
    raw = (result.stdout or "").strip()
    if not raw:
        return "(remote log empty or unavailable)"
    try:
        return base64.b64decode(raw).decode("utf-8", "replace").strip()
    except Exception:
        return "(could not decode remote log)"


def generate_on_ct(ctid: int, src: str, python: str, ref: str,
                   reps: int, with_floer: bool, timeout: int) -> str:
    remote = build_remote_command(src, python, ref, reps, with_floer)
    note("running generator on CT %d (ref %s, reps %d%s)..."
         % (ctid, ref, reps, ", with floer grid" if with_floer else ""))
    result = ct_exec(ctid, remote, timeout)

    text = result.stdout or ""
    if BEGIN in text and END in text:
        b64 = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
        try:
            markdown = base64.b64decode(b64).decode("utf-8")
        except Exception as error:
            die("could not decode the artifact returned from CT %d (%s). See output above."
                % (ctid, error))
        return markdown if markdown.endswith("\n") else markdown + "\n"

    # Did not reach extraction: surface the generator's own log so the cause is visible.
    note("--- CT %d stderr ---\n%s" % (ctid, (result.stderr or "").strip()))
    note("--- remote generator log (%s) ---\n%s" % (REMOTE_LOG, fetch_remote_log(ctid)))
    die("CT %d generation did not produce the artifact (exit %d). See output above."
        % (ctid, result.returncode))


# --- commit / push --------------------------------------------------------------------------

def compare_url(branch: str) -> str | None:
    remote = git("remote", "get-url", "origin").stdout.strip()
    match = re.search(r"github\.com[:/]+([^/]+)/(.+?)(?:\.git)?$", remote)
    if not match:
        return None
    return "https://github.com/%s/%s/compare/%s?expand=1" % (match.group(1), match.group(2), branch)


def _push_with_resync(branch: str) -> bool:
    """Push the branch. If origin advanced during the CT run so the push is non-fast-forward,
    rebase our single commit onto the new tip and retry once. Returns True on success, and
    distinguishes 'origin advanced' (retry) from other failures like auth (do not retry)."""
    pushed = git("push", "-u", "origin", branch)
    if pushed.returncode == 0:
        return True
    git("fetch", "origin", branch)
    _ahead, behind = _ahead_behind(branch)
    if behind == 0:
        note("push failed and origin is not ahead -- likely auth or permissions:\n%s"
             % pushed.stderr.strip())
        return False
    note("origin/%s advanced during the run; rebasing our commit onto it and retrying." % branch)
    rebase = git("rebase", "origin/%s" % branch)
    if rebase.returncode != 0:
        git("rebase", "--abort")
        note("rebase after the run hit a conflict (origin has a newer artifact?); the commit is "
             "local. Re-run to regenerate against the new origin.\n%s"
             % (rebase.stdout + "\n" + rebase.stderr).strip())
        return False
    retry = git("push", "-u", "origin", branch)
    if retry.returncode != 0:
        note("push still failed after rebasing onto origin:\n" + retry.stderr.strip())
        return False
    return True


def commit_and_push(out_path: str, branch: str, ctid: int, message: str, push: bool) -> None:
    rel = os.path.relpath(out_path, REPO_ROOT)
    add = git("add", "--", rel)
    if add.returncode != 0:
        die("git add failed:\n" + add.stderr.strip())

    if git("diff", "--cached", "--quiet", "--", rel).returncode == 0:
        note("%s is unchanged; nothing to commit." % rel)
        return

    if not message:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        message = "benchmarks: regenerate %s (CT %d, %s)" % (rel, ctid, stamp)
    commit = git("commit", "-m", message)
    if commit.returncode != 0:
        die("git commit failed:\n" + (commit.stderr.strip() + "\n" + commit.stdout.strip()).strip())
    note("committed: %s" % message)

    if not push:
        note("--no-push set; the commit is local on '%s'." % branch)
        return

    if not _push_with_resync(branch):
        die("git push failed; the commit is local on '%s'. Configure git credentials/remote "
            "and re-run, or push manually." % branch)
    note("pushed '%s' to origin." % branch)
    url = compare_url(branch)
    if url:
        note("open a PR: %s" % url)


# --- entry ----------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate BENCHMARKS.md on CT 250 and commit it (run on Toshiro).")
    parser.add_argument("--ctid", type=int, default=250, help="container id (default 250)")
    parser.add_argument("--ref", default="claude",
                        help="branch CT 250 is hard-reset to before generating; must hold the generator")
    parser.add_argument("--src", default="/opt/tetradrome/src", help="repo root on CT 250")
    parser.add_argument("--python", default="/opt/tetradrome/venv/bin/python",
                        help="python interpreter on CT 250")
    parser.add_argument("--reps", type=int, default=3, help="best-of-N timing repeats")
    parser.add_argument("--with-floer-grid", action="store_true",
                        help="also time the multi-core grid Floer engine")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "BENCHMARKS.md"),
                        help="local artifact path to write and commit")
    parser.add_argument("--remote-timeout", type=int, default=3600,
                        help="seconds to wait for the CT run (default 3600)")
    parser.add_argument("--message", default="", help="commit message override")
    parser.add_argument("--no-push", action="store_true", help="commit locally but do not push")
    args = parser.parse_args(argv)

    branch = preflight_git()
    sync_with_remote(branch)
    markdown = generate_on_ct(args.ctid, args.src, args.python, args.ref,
                              args.reps, args.with_floer_grid, args.remote_timeout)

    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
    note("wrote %s (%d bytes)" % (args.out, len(markdown.encode("utf-8"))))

    commit_and_push(args.out, branch, args.ctid, args.message, not args.no_push)
    note("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
