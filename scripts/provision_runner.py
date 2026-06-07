#!/usr/bin/env python3
"""Provision a lightweight Proxmox LXC, install a Python repo into it, and smoke-test it.

Run this ON a Proxmox host as root. It creates an unprivileged Debian LXC (pure compute --
no Docker, nesting, or GPU), clones a public git repo, installs it into a venv with whatever
extras you name, optionally runs a smoke-test command, and prints how to reach it. Everything
environment-specific -- the repo, the storage pool, the size, the bridge -- is a parameter,
so nothing about a particular cluster is baked in.

Required: --repo (git URL) and --rootfs-storage (a Proxmox storage pool that holds container
rootfs). The rest have generic defaults; size it up for a real workload.

    python3 scripts/provision_runner.py \
        --repo https://github.com/you/yourrepo.git --branch main \
        --rootfs-storage your-pool --cores 16 --memory 16384 \
        --extras accel --smoke "scripts/bench_grid_floer.py --knots 3_1 --sizes 5"

Re-running on an existing CTID refuses unless --recreate is given (no silent clobber).
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys

DEFAULT_TEMPLATE = "debian-12-standard_12.12-1_amd64.tar.zst"


def run(cmd: str) -> None:
    """Run a host command, echoing it; raise on failure (no silent fallbacks)."""
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def capture(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def container_exists(ctid: int) -> bool:
    return capture(f"pct status {ctid}").returncode == 0


def exec_in(ctid: int, script: str) -> None:
    """Run a bash script inside the container, fail-loud (set -euo pipefail)."""
    body = "set -euo pipefail\n" + script
    run(f"pct exec {ctid} -- bash -lc {shlex.quote(body)}")


def project_name(repo_url: str) -> str:
    """A safe install-directory name derived from the repo URL's basename."""
    base = os.path.basename(repo_url.rstrip("/")).removesuffix(".git")
    name = re.sub(r"[^A-Za-z0-9_.-]", "", base)
    if not name:
        sys.exit(f"could not derive a project name from --repo {repo_url!r}.")
    return name


def preflight() -> None:
    if os.geteuid() != 0:
        sys.exit("Run as root on the Proxmox host (pct/pveam need root).")
    missing = [tool for tool in ("pct", "pveam") if shutil.which(tool) is None]
    if missing:
        sys.exit(f"{', '.join(missing)} not found -- run this on a Proxmox host.")


def ensure_template(template: str, template_storage: str) -> None:
    print(f"[1/6] Ensuring template {template} is cached on {template_storage}...")
    if template in capture(f"pveam list {template_storage}").stdout:
        print("  already cached.")
        return
    run("pveam update")
    run(f"pveam download {template_storage} {template}")


def create_container(args) -> None:
    print(f"[2/6] Creating LXC {args.ctid} ({args.hostname})...")
    if container_exists(args.ctid):
        if not args.recreate:
            sys.exit(
                f"Container {args.ctid} already exists. Pass --recreate to destroy and rebuild "
                f"it, or choose another --ctid."
            )
        print(f"  --recreate: stopping and destroying existing {args.ctid}...")
        if "status: running" in capture(f"pct status {args.ctid}").stdout:
            run(f"pct stop {args.ctid}")
        run(f"pct destroy {args.ctid}")

    rootfs = f"{args.rootfs_storage}:{args.rootfs_size}"
    net = f"name=eth0,bridge={args.bridge},ip=dhcp"
    run(
        f"pct create {args.ctid} {args.template_storage}:vztmpl/{args.template} "
        f"--hostname {shlex.quote(args.hostname)} "
        f"--cores {args.cores} --memory {args.memory} --swap {args.swap} "
        f"--rootfs {rootfs} --net0 {net} "
        f"--unprivileged 1 --onboot 0 --tags {shlex.quote(args.tags)} --start 1"
    )


def wait_for_network(ctid: int) -> None:
    print("[3/6] Waiting for container network...")
    exec_in(ctid, 'for i in $(seq 1 30); do '
                  'getent hosts github.com >/dev/null 2>&1 && exit 0; sleep 2; done; '
                  'echo "no network in container after 60s" >&2; exit 1')


def install_repo(ctid: int, args, name: str) -> None:
    extras = f"[{args.extras}]" if args.extras else ""
    print(f"[4/6] Installing {args.repo} ({args.branch}{', extras: ' + args.extras if args.extras else ''})...")
    packages = "git python3 python3-venv python3-pip numactl ca-certificates"
    if args.apt:
        packages += " " + args.apt
    base = f"/opt/{name}"
    exec_in(ctid,
            "export DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8\n"
            "apt-get update -qq\n"
            f"apt-get install -y -qq {packages}\n"
            f"rm -rf {base}\n"
            f"python3 -m venv {base}/venv\n"
            f"{base}/venv/bin/pip install --upgrade pip -q\n"
            f"git clone --depth 1 --branch {shlex.quote(args.branch)} "
            f"{shlex.quote(args.repo)} {base}/src\n"
            f"{base}/venv/bin/pip install -q -e {shlex.quote(base + '/src' + extras)}")


def smoke_test(ctid: int, args, name: str) -> None:
    if not args.smoke:
        print("[5/6] No --smoke command given; skipping smoke test.")
        return
    print(f"[5/6] Smoke test: {args.smoke}")
    base = f"/opt/{name}"
    exec_in(ctid, f"cd {base}/src && {base}/venv/bin/python {args.smoke}")


def report(ctid: int, name: str) -> None:
    ip = capture(f"pct exec {ctid} -- hostname -I").stdout.strip().split()[:1]
    base = f"/opt/{name}"
    print("\n[6/6] Done.")
    print(f"  Container {ctid} is up{(' at ' + ip[0]) if ip else ''}.")
    print(f"  Repo at {base}/src, venv python at {base}/venv/bin/python.")
    print("\n  Run a command inside:")
    print(f"    pct exec {ctid} -- {base}/venv/bin/python {base}/src/<script> <args>")
    print(f"  Update the code later:")
    print(f"    pct exec {ctid} -- git -C {base}/src pull")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True, help="git URL to clone (public)")
    parser.add_argument("--rootfs-storage", required=True,
                        help="Proxmox storage pool for the container rootfs")
    parser.add_argument("--branch", default="main", help="repo branch (default main)")
    parser.add_argument("--extras", default="",
                        help="pip extras to install, comma-separated (e.g. accel)")
    parser.add_argument("--smoke", default="",
                        help="command (run by the venv python in the repo dir) to smoke-test")
    parser.add_argument("--apt", default="",
                        help="extra apt packages beyond the base (e.g. build-essential)")
    parser.add_argument("--ctid", type=int, default=250, help="container ID (default 250)")
    parser.add_argument("--hostname", default="lxc-runner")
    parser.add_argument("--cores", type=int, default=4, help="vCPUs (default 4)")
    parser.add_argument("--memory", type=int, default=4096, help="RAM in MiB (default 4096)")
    parser.add_argument("--swap", type=int, default=512, help="swap in MiB (default 512)")
    parser.add_argument("--rootfs-size", type=int, default=16, help="rootfs size in GiB")
    parser.add_argument("--template-storage", default="local", help="template storage")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--bridge", default="vmbr0")
    parser.add_argument("--tags", default="compute", help="Proxmox tags (semicolon-separated)")
    parser.add_argument("--recreate", action="store_true",
                        help="destroy and rebuild if the CTID already exists")
    args = parser.parse_args()

    name = project_name(args.repo)
    preflight()
    ensure_template(args.template, args.template_storage)
    create_container(args)
    wait_for_network(args.ctid)
    install_repo(args.ctid, args, name)
    smoke_test(args.ctid, args, name)
    report(args.ctid, name)


if __name__ == "__main__":
    main()
