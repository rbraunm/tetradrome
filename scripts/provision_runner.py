#!/usr/bin/env python3
"""Provision a lightweight Proxmox LXC to run the Tetradrome grid scaling sweep.

Run this from anywhere; it drives a Proxmox node over SSH (--host root@<node>, required) and
never runs against a local node. SSH is spoken directly by a pure-Python client (paramiko),
not by shelling out to ssh(1): the node password is prompted once via getpass and held only
in memory for the auth handshake -- it never reaches a command line, file, or log. One
authenticated connection is opened and reused for every command.

It creates an unprivileged Debian LXC (pure compute -- no Docker, nesting, or GPU), installs
Tetradrome into a venv with the numpy acceleration extra, smoke-tests a tiny sweep, and
prints how to run the full sweep at the container's core count with NUMA pinning.

The project is baked in (this script ships with it); only the *host* environment is
parameterized -- which node, storage pool, container size, network -- so nothing about a
particular cluster is assumed. Point --rootfs-storage at a pool that holds container rootfs,
and size --cores / --memory for the run.

    python scripts/provision_runner.py --host root@node --rootfs-storage your-pool \
        --cores 16 --memory 16384

Re-running on an existing CTID refuses unless --recreate is given (no silent clobber).
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import shlex
import sys
from collections import namedtuple

try:
    import paramiko
except ImportError:
    sys.exit(
        "provision_runner needs paramiko (the pure-Python SSH client).\n"
        "Install it from an admin prompt: pip install 'tetradrome[provision]'  (or: pip install paramiko)"
    )

DEFAULT_TEMPLATE = "debian-12-standard_12.12-1_amd64.tar.zst"
DEFAULT_REPO = "https://github.com/rbraunm/tetradrome.git"
DEFAULT_SMOKE = "scripts/bench_grid_floer.py --knots 3_1 --sizes 5"

SSH_TARGET = ""    # the user@host spec, kept for display in messages
_CLIENT = None     # the single authenticated paramiko connection, opened in connect()

# capture() mirrors the (returncode, stdout, stderr) shape callers relied on before.
Result = namedtuple("Result", "returncode stdout stderr")


def connect(host_spec: str) -> "paramiko.SSHClient":
    """Open the one SSH connection every command reuses. Password is prompted here and
    held only in memory for the handshake -- never echoed, written, or logged."""
    user, sep, hostpart = host_spec.partition("@")
    if not sep or not hostpart:
        sys.exit(f"--host must be user@host (e.g. root@labradorite), got {host_spec!r}.")
    host, _, port_s = hostpart.partition(":")
    try:
        port = int(port_s) if port_s else 22
    except ValueError:
        sys.exit(f"bad port in --host {host_spec!r}.")
    password = getpass.getpass(f"Password for {host_spec}: ")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username=user, password=password,
                       look_for_keys=False, allow_agent=False)
    except paramiko.AuthenticationException:
        sys.exit(f"authentication failed for {host_spec} (wrong password?).")
    except (paramiko.SSHException, OSError) as e:
        sys.exit(f"cannot reach {host_spec} over SSH: {e}")
    finally:
        del password
    return client


def run(cmd: str) -> None:
    """Run a Proxmox-host command over the shared connection, streaming output live;
    exit on nonzero status (no silent fallbacks)."""
    print(f"$ {cmd}")
    chan = _CLIENT.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(cmd)
    stdout = chan.makefile("r")
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = chan.recv_exit_status()
    if rc != 0:
        sys.exit(f"command failed (exit {rc}) on {SSH_TARGET}: {cmd}")


def capture(cmd: str) -> Result:
    """Run a command over the shared connection and return its full output, no echo."""
    chan = _CLIENT.get_transport().open_session()
    chan.exec_command(cmd)
    out = chan.makefile("rb").read().decode(errors="replace")
    err = chan.makefile_stderr("rb").read().decode(errors="replace")
    rc = chan.recv_exit_status()
    return Result(rc, out, err)


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
    # connect() already proved we can authenticate; confirm we landed as root on a Proxmox node.
    if capture("id -u").stdout.strip() != "0":
        sys.exit(f"remote user on {SSH_TARGET} is not root (pct/pveam need root).")
    if capture("command -v pct pveam").returncode != 0:
        sys.exit(f"{SSH_TARGET} is missing pct/pveam -- is it a Proxmox node?")


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

    if args.ip != "dhcp" and not args.gateway:
        sys.exit("A static --ip needs --gateway (the container would have no default route).")
    rootfs = f"{args.rootfs_storage}:{args.rootfs_size}"
    net = f"name=eth0,bridge={args.bridge},ip={args.ip}"
    if args.ip != "dhcp":
        net += f",gw={args.gateway}"
    if args.vlan:
        net += f",tag={args.vlan}"
    dns = ""
    if args.nameserver:
        dns += f" --nameserver {shlex.quote(args.nameserver)}"
    if args.searchdomain:
        dns += f" --searchdomain {shlex.quote(args.searchdomain)}"
    run(
        f"pct create {args.ctid} {args.template_storage}:vztmpl/{args.template} "
        f"--hostname {shlex.quote(args.hostname)} "
        f"--cores {args.cores} --memory {args.memory} --swap {args.swap} "
        f"--rootfs {rootfs} --net0 {shlex.quote(net)}{dns} "
        f"--unprivileged 1 --onboot 0 --tags {shlex.quote(args.tags)} --start 1"
    )


def wait_for_network(ctid: int) -> None:
    print("[3/6] Waiting for container network...")
    exec_in(ctid, 'for i in $(seq 1 30); do '
                  'getent hosts github.com >/dev/null 2>&1 && exit 0; sleep 2; done; '
                  'echo "no network in container after 60s" >&2; exit 1')


def install_repo(ctid: int, args, name: str) -> None:
    extras = f"[{args.extras}]" if args.extras else ""
    print(f"[4/6] Installing {args.repo} ({args.branch}"
          f"{', extras: ' + args.extras if args.extras else ''})...")
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


def report(ctid: int, args, name: str) -> None:
    ip = capture(f"pct exec {ctid} -- hostname -I").stdout.strip().split()[:1]
    base = f"/opt/{name}"
    python = f"{base}/venv/bin/python"
    bench = f"{base}/src/scripts/bench_grid_floer.py"
    print("\n[6/6] Done.")
    print(f"  Container {ctid} is up{(' at ' + ip[0]) if ip else ''} on {SSH_TARGET}.")
    print(f"  Repo at {base}/src, venv python at {python}.")
    print(f"  (run the commands below on {SSH_TARGET}, or prefix each with: ssh {SSH_TARGET})")
    print("\n  Run the scaling sweep (synthetic sizes isolate generation; --pin needs Linux):")
    print(f"    pct exec {ctid} -- {python} {bench} \\")
    print(f"        --sizes 8 9 10 11 --gen-workers {args.cores} --workers {args.cores} --pin")
    print(f"\n  Update the code later:")
    print(f"    pct exec {ctid} -- git -C {base}/src pull")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Host environment -- parameterized, no cluster assumptions:
    parser.add_argument("--host", required=True,
                        help="Proxmox node to drive over SSH (e.g. root@labradorite)")
    parser.add_argument("--rootfs-storage", default="local-lvm",
                        help="Proxmox storage pool for the container rootfs (default local-lvm)")
    parser.add_argument("--ctid", type=int, default=250, help="container ID (default 250)")
    parser.add_argument("--hostname", default="tetradrome", help="container hostname")
    parser.add_argument("--cores", type=int, default=4, help="vCPUs (default 4; size up)")
    parser.add_argument("--memory", type=int, default=4096, help="RAM in MiB (default 4096)")
    parser.add_argument("--swap", type=int, default=512, help="swap in MiB (default 512)")
    parser.add_argument("--rootfs-size", type=int, default=16, help="rootfs size in GiB")
    parser.add_argument("--template-storage", default="local", help="template storage")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--bridge", default="vmbr0", help="network bridge (default vmbr0)")
    parser.add_argument("--ip", default="dhcp",
                        help="container IPv4: 'dhcp' or CIDR like 10.0.0.5/24 (default dhcp)")
    parser.add_argument("--gateway", default="", help="default gateway (required with a static --ip)")
    parser.add_argument("--vlan", type=int, default=0, help="VLAN tag for the NIC (0 = untagged)")
    parser.add_argument("--nameserver", default="", help="DNS server(s) (default: from DHCP)")
    parser.add_argument("--searchdomain", default="", help="DNS search domain (default: from DHCP)")
    parser.add_argument("--tags", default="tetradrome;compute",
                        help="Proxmox tags (semicolon-separated)")
    parser.add_argument("--recreate", action="store_true",
                        help="destroy and rebuild if the CTID already exists")
    # Project -- baked in, overridable:
    parser.add_argument("--repo", default=DEFAULT_REPO, help="git URL to clone")
    parser.add_argument("--branch", default="main", help="repo branch (default main)")
    parser.add_argument("--extras", default="accel",
                        help="pip extras, comma-separated (default accel; numpy reducer)")
    parser.add_argument("--smoke", default=DEFAULT_SMOKE,
                        help="command (venv python, in the repo dir) to smoke-test; '' to skip")
    parser.add_argument("--apt", default="",
                        help="extra apt packages beyond the base")
    args = parser.parse_args()

    global SSH_TARGET, _CLIENT
    SSH_TARGET = args.host
    _CLIENT = connect(args.host)
    try:
        name = project_name(args.repo)
        preflight()
        ensure_template(args.template, args.template_storage)
        create_container(args)
        wait_for_network(args.ctid)
        install_repo(args.ctid, args, name)
        smoke_test(args.ctid, args, name)
        report(args.ctid, args, name)
    finally:
        _CLIENT.close()


if __name__ == "__main__":
    main()
