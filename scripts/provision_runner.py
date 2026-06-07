#!/usr/bin/env python3
"""Provision a lightweight Proxmox LXC to run the Tetradrome grid scaling sweep.

Run this ON the Proxmox host (e.g. labradorite) as root. It creates an unprivileged Debian
LXC, installs Python + Tetradrome (the public repo, into a venv) with the numpy acceleration
extra, and leaves it ready to run ``scripts/bench_grid_floer.py`` -- with the full core count
and NUMA pinning that only mean anything on a multi-socket host like this one.

Pure compute: no Docker, no nesting, no GPU. Modeled on ``provision-dragonglass.sh`` for the
Proxmox-side specifics (Debian template, ``pct create`` flags, ``alpha`` storage, ``vmbr0``
bridge), rewritten in Python for portability and made fail-loud and non-interactive.

    python3 scripts/provision_runner.py --cores 64 --memory 32768
    # then, inside the container (the script prints the exact lines):
    /opt/tetradrome/venv/bin/python /opt/tetradrome/src/scripts/bench_grid_floer.py \
        --sizes 8 9 10 11 --gen-workers 88 --workers 88 --pin

Re-running on an existing CTID refuses unless --recreate is given (no silent clobber).
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

DEFAULT_TEMPLATE = "debian-12-standard_12.12-1_amd64.tar.zst"
REPO_URL = "https://github.com/rbraunm/tetradrome.git"


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


def preflight() -> None:
    if os.geteuid() != 0:
        sys.exit("Run as root on the Proxmox host (pct/pveam need root).")
    missing = [tool for tool in ("pct", "pveam") if shutil.which(tool) is None]
    if missing:
        sys.exit(f"{', '.join(missing)} not found -- run this on a Proxmox host (labradorite).")


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
        f"--unprivileged 1 --onboot 0 --tags {shlex.quote('tetradrome;compute')} --start 1"
    )


def wait_for_network(ctid: int) -> None:
    print("[3/6] Waiting for container network...")
    exec_in(ctid, 'for i in $(seq 1 30); do '
                  'getent hosts github.com >/dev/null 2>&1 && exit 0; sleep 2; done; '
                  'echo "no network in container after 60s" >&2; exit 1')


def install_tetradrome(ctid: int, branch: str, with_numba: bool) -> None:
    extras = "accel,jit" if with_numba else "accel"
    print(f"[4/6] Installing Tetradrome ({branch}, extras: {extras}) into the container...")
    packages = "git python3 python3-venv python3-pip numactl ca-certificates"
    if with_numba:
        packages += " build-essential"          # numba/llvmlite may need a compiler
    exec_in(ctid,
            "export DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8\n"
            "apt-get update -qq\n"
            f"apt-get install -y -qq {packages}\n"
            "rm -rf /opt/tetradrome\n"
            "python3 -m venv /opt/tetradrome/venv\n"
            "/opt/tetradrome/venv/bin/pip install --upgrade pip -q\n"
            f"git clone --depth 1 --branch {shlex.quote(branch)} {REPO_URL} /opt/tetradrome/src\n"
            f"/opt/tetradrome/venv/bin/pip install -q -e '/opt/tetradrome/src[{extras}]'")


def smoke_test(ctid: int) -> None:
    print("[5/6] Smoke test: a tiny sweep inside the container...")
    exec_in(ctid, "/opt/tetradrome/venv/bin/python "
                  "/opt/tetradrome/src/scripts/bench_grid_floer.py --knots 3_1 --sizes 5")


def report(ctid: int) -> None:
    ip = capture(f"pct exec {ctid} -- hostname -I").stdout.strip().split()[:1]
    print("\n[6/6] Done.")
    print(f"  Container {ctid} is up{(' at ' + ip[0]) if ip else ''}.")
    py = "/opt/tetradrome/venv/bin/python"
    bench = "/opt/tetradrome/src/scripts/bench_grid_floer.py"
    print("\n  Run the scaling sweep:")
    print(f"    pct exec {ctid} -- {py} {bench} \\")
    print("        --sizes 8 9 10 11 --gen-workers 88 --workers 88 --pin")
    print(f"\n    pct exec {ctid} -- {py} {bench} \\")
    print("        --knots 3_1 5_2 8_19 6_1 7_1 --gen-workers 88 --workers 88 --pin")
    print(f"\n  To update the code later: "
          f"pct exec {ctid} -- git -C /opt/tetradrome/src pull")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ctid", type=int, default=250, help="container ID (default 250)")
    parser.add_argument("--hostname", default="tetradrome")
    parser.add_argument("--cores", type=int, default=64, help="vCPUs (default 64)")
    parser.add_argument("--memory", type=int, default=32768, help="RAM in MiB (default 32768)")
    parser.add_argument("--swap", type=int, default=4096, help="swap in MiB (default 4096)")
    parser.add_argument("--rootfs-storage", default="alpha", help="rootfs storage pool")
    parser.add_argument("--rootfs-size", type=int, default=16, help="rootfs size in GiB")
    parser.add_argument("--template-storage", default="local", help="template storage")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--bridge", default="vmbr0")
    parser.add_argument("--branch", default="claude", help="repo branch to install")
    parser.add_argument("--with-numba", action="store_true",
                        help="also install the numba jit reducer tier (heavier)")
    parser.add_argument("--recreate", action="store_true",
                        help="destroy and rebuild if the CTID already exists")
    args = parser.parse_args()

    preflight()
    ensure_template(args.template, args.template_storage)
    create_container(args)
    wait_for_network(args.ctid)
    install_tetradrome(args.ctid, args.branch, args.with_numba)
    smoke_test(args.ctid)
    report(args.ctid)


if __name__ == "__main__":
    main()
