# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Integration tests for tools/generate_benchmarks.py git-safety.

Each test stands up a throwaway bare "origin" and one or two working clones (no network), points
the script's git() at a working clone by setting REPO_ROOT, and drives the real sync / push
helpers through the cases the artifact push actually hits: up to date, behind, diverged, remote
branch absent, a push race, and a conflicting race. git is the only dependency.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_benchmarks as gb  # noqa: E402


def run(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def out(cwd, *args):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def commit_file(repo, name, content, message):
    with open(os.path.join(repo, name), "w") as handle:
        handle.write(content)
    run(repo, "git", "add", "--", name)
    run(repo, "git", "commit", "-m", message)


def make_origin_and_clone(tmp):
    """A bare origin holding branches main and claude, plus a working clone checked out on claude."""
    origin = os.path.join(tmp, "origin.git")
    run(tmp, "git", "init", "--bare", origin)
    work = os.path.join(tmp, "work")
    run(tmp, "git", "clone", origin, work)
    run(work, "git", "config", "user.email", "t@example.com")
    run(work, "git", "config", "user.name", "Tester")
    commit_file(work, "README", "seed\n", "init")
    run(work, "git", "branch", "-M", "main")
    run(work, "git", "push", "-u", "origin", "main")
    run(work, "git", "checkout", "-b", "claude")
    run(work, "git", "push", "-u", "origin", "claude")
    return origin, work


def clone_and_advance(tmp, origin, branch, name, content):
    """A second clone that pushes one commit to origin/<branch>, so origin moves ahead."""
    other = os.path.join(tmp, "other-" + name)
    run(tmp, "git", "clone", origin, other)
    run(other, "git", "config", "user.email", "o@example.com")
    run(other, "git", "config", "user.name", "Other")
    run(other, "git", "checkout", branch)
    commit_file(other, name, content, "advance " + name)
    run(other, "git", "push", "origin", branch)
    return other


def remote_sha(repo, branch):
    return out(repo, "git", "ls-remote", "origin", "refs/heads/" + branch).split()[0]


def is_ancestor(repo, maybe_ancestor, ref):
    return subprocess.run(["git", "merge-base", "--is-ancestor", maybe_ancestor, ref],
                          cwd=repo).returncode == 0


def test_sync_up_to_date_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        _origin, work = make_origin_and_clone(tmp)
        before = out(work, "git", "rev-parse", "HEAD")
        gb.REPO_ROOT = work
        gb.sync_with_remote("claude")
        assert out(work, "git", "rev-parse", "HEAD") == before


def test_sync_behind_fast_forwards():
    with tempfile.TemporaryDirectory() as tmp:
        origin, work = make_origin_and_clone(tmp)
        clone_and_advance(tmp, origin, "claude", "a.txt", "1\n")
        gb.REPO_ROOT = work
        gb.sync_with_remote("claude")
        assert out(work, "git", "rev-parse", "HEAD") == remote_sha(work, "claude")
        assert os.path.exists(os.path.join(work, "a.txt"))


def test_sync_diverged_rebases():
    with tempfile.TemporaryDirectory() as tmp:
        origin, work = make_origin_and_clone(tmp)
        commit_file(work, "local.txt", "L\n", "local unpushed work")     # local ahead by 1
        clone_and_advance(tmp, origin, "claude", "remote.txt", "R\n")    # origin ahead by 1
        gb.REPO_ROOT = work
        gb.sync_with_remote("claude")
        # our unpushed commit is replayed on top of origin's tip: both files present, origin is
        # now an ancestor of HEAD, and HEAD is one commit ahead of origin.
        assert os.path.exists(os.path.join(work, "local.txt"))
        assert os.path.exists(os.path.join(work, "remote.txt"))
        assert is_ancestor(work, "origin/claude", "HEAD")
        assert out(work, "git", "rev-list", "--count", "origin/claude..HEAD") == "1"


def test_sync_absent_remote_branch_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        _origin, work = make_origin_and_clone(tmp)
        run(work, "git", "checkout", "-b", "feature-x")                  # never pushed
        before = out(work, "git", "rev-parse", "HEAD")
        gb.REPO_ROOT = work
        gb.sync_with_remote("feature-x")                                 # must not raise
        assert out(work, "git", "rev-parse", "HEAD") == before


def test_push_with_resync_retries_on_race():
    with tempfile.TemporaryDirectory() as tmp:
        origin, work = make_origin_and_clone(tmp)
        commit_file(work, "artifact.txt", "A\n", "artifact")             # our commit to push
        clone_and_advance(tmp, origin, "claude", "other.txt", "O\n")     # origin raced ahead
        gb.REPO_ROOT = work
        assert gb._push_with_resync("claude") is True
        head = out(work, "git", "rev-parse", "HEAD")
        assert remote_sha(work, "claude") == head                        # push landed (ff)
        assert os.path.exists(os.path.join(work, "artifact.txt"))
        assert os.path.exists(os.path.join(work, "other.txt"))           # rebased onto the race


def test_push_with_resync_returns_false_on_conflicting_race():
    with tempfile.TemporaryDirectory() as tmp:
        origin, work = make_origin_and_clone(tmp)
        commit_file(work, "artifact.txt", "ours\n", "artifact ours")
        clone_and_advance(tmp, origin, "claude", "artifact.txt", "theirs\n")   # same file, conflict
        gb.REPO_ROOT = work
        assert gb._push_with_resync("claude") is False
        # rebase aborted: our commit is intact and unpushed, working tree not left mid-rebase.
        with open(os.path.join(work, "artifact.txt")) as handle:
            assert handle.read() == "ours\n"
        assert not os.path.isdir(os.path.join(work, ".git", "rebase-merge"))


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
