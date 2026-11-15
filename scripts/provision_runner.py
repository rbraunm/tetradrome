#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Provision a Proxmox LXC sized and pinned for Tetradrome's compute runs.

Run this from anywhere; it drives a Proxmox node over SSH and never runs against a local node.

    python scripts/provision_runner.py --host root@node --rootfs-storage your-pool \
        --cores 16 --memory 32768

Logging in. --host is any login on the node: root, or an account with passwordless sudo (every
node command then runs through `sudo -n`). The password is prompted once via getpass and held
only in memory for the auth handshake -- it never reaches a command line, file, or log. SSH is
spoken by a pure-Python client (paramiko), and one authenticated connection is reused for every
command. Automation that already holds an open paramiko connection to the node calls
provision(args, client) instead of main() and skips the prompt.

Sizing. --cores and --memory are required: the right size depends on the node, and a box that is
quietly too small cannot run the heavy tier. --cores counts logical CPUs. The runner reads the
node's NUMA and hyperthread layout and gives the container an explicit CPU list -- whole physical
cores, the same number from every NUMA node -- because the scheduler places work per NUMA node
and counts every CPU it is allowed as a core, while Proxmox left alone picks CPUs without regard
to sockets and reshuffles them as other containers come and go. --memory is refused when it
exceeds what the node has available.

What it builds. An unprivileged Debian LXC (pure compute -- no Docker, nesting, or GPU) with
Tetradrome in a venv and the full runnable suite by default (every CPU acceleration tier, the
KnotInfo backend, and pytest), the comparison oracles (SageMath plus the built gold-masters
KnotJob, JavaKh, kht++, knotkit, and KhoHo) unless --skip-oracles, sshd with a dedicated login,
and its hostname advertised as <name>.local over mDNS (Avahi). It ends with a tiny smoke sweep and
prints how to run the real ones.

The login. --ssh-user (default 'tetradrome') owns the install. Each provision generates, fresh, a
password and an ed25519 keypair. The private key is generated on this controller and never leaves
it -- only the public line is pushed to the container -- and is written chmod-600 beside this
script (gitignored), where tools/ct_exec.py finds it. The address, user, and password go to a
chmod-600 file beside it. The login is created before the long installs, so a failed build still
leaves a box you can reach.

Re-running on an existing CTID refuses unless --recreate is given (no silent clobber).
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import shlex
import sys
from collections import namedtuple

try:
    import paramiko
    # paramiko has no ed25519 generator, so the keypair is built with cryptography, which it
    # always brings along.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    sys.exit(
        "provision_runner needs paramiko (the pure-Python SSH client).\n"
        "Install it from an admin prompt: pip install 'tetradrome[provision]'  (or: pip install paramiko)"
    )

DEFAULT_TEMPLATE = "debian-12-standard_12.12-1_amd64.tar.zst"
DEFAULT_REPO = "https://github.com/rbraunm/tetradrome.git"
DEFAULT_SMOKE = "scripts/bench_grid_floer.py --knots 3_1 --sizes 5"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PACKAGES = ("git curl gnupg python3 python3-venv python3-pip numactl ca-certificates "
                 "openssh-server avahi-daemon libnss-mdns")

# One line per fact, each tagged, so parse_topology never depends on output order.
TOPOLOGY_PROBE = (
    'for n in /sys/devices/system/node/node[0-9]*; do '
    'echo "node ${n##*node} $(cat "$n/cpulist")"; done; '
    'for c in /sys/devices/system/cpu/cpu[0-9]*; do '
    'if [ -r "$c/topology/thread_siblings_list" ]; then '
    'echo "cpu ${c##*cpu} $(cat "$c/topology/thread_siblings_list")"; fi; done; '
    "awk '/^MemAvailable:/ {print \"memavailable\", $2}' /proc/meminfo; "
    'echo "cgroup $(stat -fc %T /sys/fs/cgroup)"'
)

Result = namedtuple("Result", "returncode stdout stderr")
Topology = namedtuple("Topology", "nodes siblings mem_available_mib cgroup")
Login = namedtuple("Login", "ctid hostname node address mdns_name user password key_path repo python")


class ProvisionError(Exception):
    pass


# ---- sizing: pure functions over the node's reported layout ----------------------------

def parse_cpulist(text: str) -> list[int]:
    """A Linux cpulist like '0-3,8' as sorted CPU ids."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return sorted(cpus)


def format_cpulist(cpus) -> str:
    """CPU ids as a compact cpulist: [28, 29, 30, 72] -> '28-30,72'."""
    ranges: list[list[int]] = []
    for cpu in sorted(cpus):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(low) if low == high else f"{low}-{high}" for low, high in ranges)


def parse_topology(text: str) -> Topology:
    """The node's NUMA nodes, hyperthread siblings, available memory, and cgroup version, from
    TOPOLOGY_PROBE's output. A NUMA node with no CPUs (memory only) is left out."""
    nodes: dict[int, list[int]] = {}
    siblings: dict[int, tuple[int, ...]] = {}
    mem_available_mib = None
    cgroup = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["node"] and len(fields) == 3:
            nodes[int(fields[1])] = parse_cpulist(fields[2])
        elif fields[:1] == ["cpu"] and len(fields) == 3:
            siblings[int(fields[1])] = tuple(parse_cpulist(fields[2]))
        elif fields[:1] == ["memavailable"] and len(fields) == 2:
            mem_available_mib = int(fields[1]) // 1024
        elif fields[:1] == ["cgroup"] and len(fields) == 2:
            cgroup = fields[1]
    unmapped = sorted(cpu for cpus in nodes.values() for cpu in cpus if cpu not in siblings)
    if not nodes or unmapped or mem_available_mib is None or cgroup is None:
        raise ProvisionError("could not read the node's CPU and memory layout from:\n" + text)
    return Topology(nodes, siblings, mem_available_mib, cgroup)


def select_cpus(topology: Topology, count: int) -> list[int]:
    """`count` logical CPUs as whole physical cores, the same number from every NUMA node."""
    cores_by_node = {
        index: sorted({topology.siblings[cpu] for cpu in cpus})
        for index, cpus in topology.nodes.items()
    }
    widths = {len(core) for cores in cores_by_node.values() for core in cores}
    if len(widths) != 1:
        raise ProvisionError(f"the node mixes cores of {sorted(widths)} threads; cannot split it evenly.")
    threads = widths.pop()
    step = len(cores_by_node) * threads
    if count <= 0 or count % step:
        raise ProvisionError(
            f"--cores {count} must be a positive multiple of {step}: {len(cores_by_node)} NUMA "
            f"node(s) x {threads} thread(s) per core, so every node gives the same number of whole cores.")
    wanted = count // step
    smallest = min(len(cores) for cores in cores_by_node.values())
    if wanted > smallest:
        raise ProvisionError(
            f"--cores {count} needs {wanted} core(s) from each NUMA node, but the smallest node has "
            f"{smallest}; the most this node can give is {smallest * step}.")
    # From the top of each node: CPU 0 is the boot CPU, and its core stays with the host longest.
    return sorted(cpu for cores in cores_by_node.values() for core in cores[-wanted:] for cpu in core)


# ---- the node: one SSH connection, every command as root -------------------------------

class Node:
    """The Proxmox node behind one open paramiko connection."""

    def __init__(self, client, label: str, verbose: bool):
        self.client = client
        self.label = label
        self.verbose = verbose
        self.sudo = False

    def _as_root(self, cmd: str) -> str:
        return f"sudo -n bash -c {shlex.quote(cmd)}" if self.sudo else cmd

    def capture(self, cmd: str, *, root: bool = True) -> Result:
        """Run a command and return its status and output, no echo."""
        chan = self.client.get_transport().open_session()
        chan.exec_command(self._as_root(cmd) if root else cmd)
        out = chan.makefile("rb").read().decode(errors="replace")
        err = chan.makefile_stderr("rb").read().decode(errors="replace")
        return Result(chan.recv_exit_status(), out, err)

    def run(self, cmd: str, *, stream: bool = False, stdin_data: str | None = None) -> None:
        """Run a command as root and raise on a nonzero status. Quiet by default: output is shown
        only if the command fails. stream=True (or --debug) shows it live. stdin_data reaches the
        command over the channel, so a secret in it never lands on a command line or in the log."""
        live = self.verbose or stream
        shown = cmd if stdin_data is None else f"{cmd}  (stdin withheld)"
        if self.verbose:
            print(f"$ {shown}")
        chan = self.client.get_transport().open_session()
        chan.set_combine_stderr(True)   # one stream to drain -> no stdout/stderr deadlock
        chan.exec_command(self._as_root(cmd))
        if stdin_data is not None:
            chan.sendall(stdin_data.encode())
            chan.shutdown_write()
        out = chan.makefile("rb")
        captured = ""
        if live:
            for line in iter(out.readline, b""):
                sys.stdout.write(line.decode(errors="replace"))
                sys.stdout.flush()
        else:
            captured = out.read().decode(errors="replace")   # drain fully before recv_exit_status
        status = chan.recv_exit_status()
        if status != 0:
            if not live:
                print(f"$ {shown}")
                sys.stdout.write(captured)
                sys.stdout.flush()
            raise ProvisionError(f"command failed (exit {status}) on {self.label}: {shown}")

    def exec_in(self, ctid: int, script: str, *, stream: bool = False) -> None:
        """Run a bash script inside the container, fail-loud (set -euo pipefail)."""
        body = "set -euo pipefail\n" + script
        self.run(f"pct exec {ctid} -- bash -lc {shlex.quote(body)}", stream=stream)


def connect(host_spec: str) -> "paramiko.SSHClient":
    """Open the connection for main(). The password is prompted here and held only in memory
    for the handshake -- never echoed, written, or logged."""
    user, sep, hostpart = host_spec.partition("@")
    if not sep or not hostpart:
        raise ProvisionError(f"--host must be user@host (e.g. root@node), got {host_spec!r}.")
    host, _, port_s = hostpart.partition(":")
    try:
        port = int(port_s) if port_s else 22
    except ValueError:
        raise ProvisionError(f"bad port in --host {host_spec!r}.") from None
    password = getpass.getpass(f"Password for {host_spec}: ")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username=user, password=password,
                       look_for_keys=False, allow_agent=False)
    except paramiko.AuthenticationException:
        raise ProvisionError(f"authentication failed for {host_spec} (wrong password?).") from None
    except (paramiko.SSHException, OSError) as e:
        raise ProvisionError(f"cannot reach {host_spec} over SSH: {e}") from e
    finally:
        del password
    return client


# ---- steps -----------------------------------------------------------------------------

def check_arguments(args) -> None:
    if args.ip != "dhcp" and not args.gateway:
        raise ProvisionError("A static --ip needs --gateway (the container would have no default route).")
    if args.hwaddr and not re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", args.hwaddr):
        raise ProvisionError(f"--hwaddr {args.hwaddr!r} is not a MAC address like BC:24:11:A9:AA:AD.")


def project_name(repo_url: str) -> str:
    """A safe install-directory name derived from the repo URL's basename."""
    base = os.path.basename(repo_url.rstrip("/")).removesuffix(".git")
    name = re.sub(r"[^A-Za-z0-9_.-]", "", base)
    if not name:
        raise ProvisionError(f"could not derive a project name from --repo {repo_url!r}.")
    return name


def preflight(node: Node) -> None:
    """Settle how commands become root, and confirm this is a Proxmox node."""
    if node.capture("id -u", root=False).stdout.strip() != "0":
        if node.capture("sudo -n true", root=False).returncode != 0:
            raise ProvisionError(
                f"{node.label} is not root and cannot sudo without a password (pct/pveam need root). "
                f"Log in as root, or give this account passwordless sudo.")
        node.sudo = True
    if node.capture("command -v pct pveam").returncode != 0:
        raise ProvisionError(f"{node.label} is missing pct/pveam -- is it a Proxmox node?")


def plan_cpus(node: Node, args) -> list[int]:
    print("[1/10] Reading the node's CPU and memory layout...")
    topology = parse_topology(node.capture(TOPOLOGY_PROBE).stdout)
    if topology.cgroup != "cgroup2fs":
        raise ProvisionError(
            f"{node.label} mounts {topology.cgroup} at /sys/fs/cgroup; pinning the container's CPUs "
            f"needs cgroup v2.")
    if args.memory > topology.mem_available_mib:
        raise ProvisionError(
            f"--memory {args.memory} MiB is more than the {topology.mem_available_mib} MiB "
            f"{node.label} has available.")
    cpus = select_cpus(topology, args.cores)
    threads = len(topology.siblings[cpus[0]])
    per_node = args.cores // (len(topology.nodes) * threads)
    print(f"  CPUs {format_cpulist(cpus)}: {per_node} core(s) x {threads} thread(s) from each of "
          f"{len(topology.nodes)} NUMA node(s).")
    print(f"  memory {args.memory} MiB of the {topology.mem_available_mib} MiB available.")
    return cpus


def ensure_template(node: Node, template: str, template_storage: str) -> None:
    print(f"[2/10] Ensuring template {template} is cached on {template_storage}...")
    if template in node.capture(f"pveam list {template_storage}").stdout:
        print("  already cached.")
        return
    node.run("pveam update")
    node.run(f"pveam download {template_storage} {template}")


def create_container(node: Node, args, cpus: list[int]) -> None:
    ctid = args.ctid
    print(f"[3/10] Creating LXC {ctid} ({args.hostname})...")
    if node.capture(f"pct status {ctid}").returncode == 0:
        if not args.recreate:
            raise ProvisionError(
                f"Container {ctid} already exists. Pass --recreate to destroy and rebuild it, or "
                f"choose another --ctid.")
        print(f"  --recreate: stopping and destroying existing {ctid}...")
        if "status: running" in node.capture(f"pct status {ctid}").stdout:
            node.run(f"pct stop {ctid}")
        node.run(f"pct destroy {ctid}")

    net = f"name=eth0,bridge={args.bridge},ip={args.ip}"
    if args.ip != "dhcp":
        net += f",gw={args.gateway}"
    if args.vlan:
        net += f",tag={args.vlan}"
    if args.hwaddr:
        net += f",hwaddr={args.hwaddr.upper()}"
    dns = ""
    if args.nameserver:
        dns += f" --nameserver {shlex.quote(args.nameserver)}"
    if args.searchdomain:
        dns += f" --searchdomain {shlex.quote(args.searchdomain)}"
    node.run(
        f"pct create {ctid} {args.template_storage}:vztmpl/{args.template} "
        f"--hostname {shlex.quote(args.hostname)} "
        f"--cores {args.cores} --memory {args.memory} --swap {args.swap} "
        f"--rootfs {args.rootfs_storage}:{args.rootfs_size} --net0 {shlex.quote(net)}{dns} "
        f"--unprivileged 1 --onboot 0 --tags {shlex.quote(args.tags)}"
    )
    # An explicit cpuset in the config is the one thing Proxmox leaves alone: without it, it
    # picks --cores CPUs at random on start and rebalances them while the container runs.
    cpulist = format_cpulist(cpus)
    node.run(f"printf 'lxc.cgroup2.cpuset.cpus: %s\\n' {cpulist} >> /etc/pve/lxc/{ctid}.conf")
    node.run(f"pct start {ctid}")

    effective = node.capture(f"cat /sys/fs/cgroup/lxc/{ctid}/cpuset.cpus.effective").stdout.strip()
    inside = node.capture(f"pct exec {ctid} -- nproc").stdout.strip()
    if not effective or parse_cpulist(effective) != cpus or inside != str(len(cpus)):
        raise ProvisionError(
            f"container {ctid} did not get the CPUs it was given: asked for {cpulist}, the node "
            f"reports {effective or 'nothing'}, and the container counts {inside or 'nothing'}.")
    print(f"  created: CT {ctid} ({args.hostname}), {args.rootfs_size} GiB rootfs on "
          f"{args.rootfs_storage}, {args.memory} MiB, CPUs {cpulist} (verified).")


def wait_for_network(node: Node, ctid: int) -> None:
    print("[4/10] Waiting for container network...")
    node.exec_in(ctid, 'for i in $(seq 1 30); do '
                       'getent hosts github.com >/dev/null 2>&1 && exit 0; sleep 2; done; '
                       'echo "no network in container after 60s" >&2; exit 1')
    print("  network up.")


def set_install_service_policy(node: Node, ctid: int) -> None:
    """Install a policy-rc.d that returns 101 so apt maintainer scripts do not (re)start
    services during package install. deb-systemd-invoke honors this, which is what otherwise
    prints the 'disabled or static unit' note and the 'Could not execute systemctl' die on an
    openssh upgrade. Every service we need is started explicitly afterward, with verification,
    and a direct systemctl call ignores policy-rc.d."""
    node.exec_in(ctid,
                 "printf '#!/bin/sh\\nexit 101\\n' > /usr/sbin/policy-rc.d\n"
                 "chmod 0755 /usr/sbin/policy-rc.d\n"
                 "test -x /usr/sbin/policy-rc.d\n")


def clear_install_service_policy(node: Node, ctid: int) -> None:
    node.exec_in(ctid, "rm -f /usr/sbin/policy-rc.d\n")


def install_base_packages(node: Node, args) -> None:
    print("[5/10] Installing base packages...")
    packages = f"{BASE_PACKAGES} {args.apt}".strip()
    node.exec_in(args.ctid,
                 "export DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8\n"
                 "apt-get update -qq\n"
                 f"apt-get install -y -qq {packages}\n")
    print("  installed.")


def generate_keypair(comment: str) -> tuple[str, str]:
    """A fresh ed25519 keypair: (OpenSSH private-key text, authorized_keys public line)."""
    key = Ed25519PrivateKey.generate()
    private_text = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_ssh = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode()
    return private_text, f"{public_ssh} {comment}"


def write_private_key(ctid: int, private_text: str, public_line: str) -> str:
    """Persist the keypair beside this script (gitignored) and return the private key's path --
    what a client passes to ssh -i."""
    priv_path = os.path.join(SCRIPT_DIR, f"ct{ctid}-ssh-key")
    # created already-restricted, so there is no world-readable window
    fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(private_text)
    os.chmod(priv_path, 0o600)
    with open(priv_path + ".pub", "w") as f:
        f.write(public_line + "\n")
    return priv_path


def write_login_file(login: Login, include_password: bool = True) -> str:
    """Write the container login to a chmod-600 file beside this script (gitignored), where
    tools/ct_exec.py reads the address and user, and return its path. A caller that keeps the
    password somewhere else passes include_password=False."""
    path = os.path.join(SCRIPT_DIR, f"ct{login.ctid}-ssh-credentials.txt")
    lines = [
        "# provision_runner.py generated container login -- DO NOT COMMIT (gitignored).",
        f"host:     {login.hostname} (CT {login.ctid}) on {login.node}",
        f"address:  {login.address or '(unknown -- run: pct exec %d -- hostname -I)' % login.ctid}",
        f"mdns:     {login.mdns_name}.local",
        f"user:     {login.user}",
    ]
    if include_password:
        lines.append(f"password: {login.password}")
    lines += [
        f"ssh-key:  {login.key_path}   (ssh -i this; key auth)",
        f"repo:     {login.repo}",
        f"venv:     {login.python}",
        "",
    ]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o600)
    return path


def setup_login(node: Node, args, name: str, save_login) -> Login:
    ctid = args.ctid
    print(f"[6/10] Enabling sshd and the {args.ssh_user!r} login...")
    user = shlex.quote(args.ssh_user)
    node.exec_in(ctid,
                 f"id -u {user} >/dev/null 2>&1 || useradd --create-home --shell /bin/bash {user}\n"
                 "install -d -m 0755 /etc/ssh/sshd_config.d\n"
                 "printf 'PasswordAuthentication yes\\nPubkeyAuthentication yes\\n'"
                 " > /etc/ssh/sshd_config.d/10-tetradrome.conf\n"
                 "systemctl enable --now ssh\n"
                 "systemctl restart ssh\n"
                 # grep without -q reads all input, so no SIGPIPE under pipefail
                 "systemctl is-active --quiet ssh\n"
                 "sshd -T 2>/dev/null | grep -i '^passwordauthentication yes'\n")
    password = secrets.token_urlsafe(18)
    private_text, public_line = generate_keypair(f"tetradrome-provisioner-ct{ctid}")
    node.run(f"pct exec {ctid} -- chpasswd", stdin_data=f"{args.ssh_user}:{password}\n")
    key_q = shlex.quote(public_line)
    node.exec_in(ctid,
                 f"user={user}\n"
                 'home=$(getent passwd "$user" | cut -d: -f6)\n'
                 '[ -n "$home" ] || { echo "no home dir for $user" >&2; exit 1; }\n'
                 'install -d -m 700 -o "$user" -g "$user" "$home/.ssh"\n'
                 f'printf "%s\\n" {key_q} > "$home/.ssh/authorized_keys"\n'
                 'chmod 600 "$home/.ssh/authorized_keys"\n'
                 'chown "$user":"$user" "$home/.ssh/authorized_keys"\n'
                 f'grep -qF {key_q} "$home/.ssh/authorized_keys"\n')
    addresses = node.capture(f"pct exec {ctid} -- hostname -I").stdout.split()
    base = f"/opt/{name}"
    login = Login(ctid=ctid, hostname=args.hostname, node=node.label,
                  address=addresses[0] if addresses else "",
                  mdns_name=args.mdns_hostname or args.hostname, user=args.ssh_user,
                  password=password, key_path=write_private_key(ctid, private_text, public_line),
                  repo=f"{base}/src", python=f"{base}/venv/bin/python")
    where = save_login(login)
    print(f"  sshd active; password + key auth enabled for {args.ssh_user!r}.")
    print(f"  private key: {login.key_path}   (chmod 600, gitignored)")
    print(f"  login recorded in: {where}")
    return login


def install_repo(node: Node, args, name: str) -> None:
    extras = f"[{args.extras}]" if args.extras else ""
    print(f"[7/10] Installing {args.repo} ({args.branch}"
          f"{', extras: ' + args.extras if args.extras else ''})...")
    base = f"/opt/{name}"
    node.exec_in(args.ctid,
                 f"rm -rf {base}\n"
                 f"python3 -m venv {base}/venv\n"
                 f"{base}/venv/bin/pip install --upgrade pip -q\n"
                 f"git clone --depth 1 --branch {shlex.quote(args.branch)} "
                 f"{shlex.quote(args.repo)} {base}/src\n"
                 f"{base}/venv/bin/pip install -q -e {shlex.quote(base + '/src' + extras)}")
    # pip exits 0 on an unknown extra (it warns and installs the bare package), so prove the
    # package imports and that its metadata defines every extra that was asked for.
    requested = [e.strip() for e in args.extras.split(",") if e.strip()]
    check = "\n".join([
        "import importlib.metadata as md, sys",
        f"import {name}" if name.isidentifier() else "",
        f"requested = {requested!r}",
        f"branch = {args.branch!r}",
        f"provided = set(md.metadata({name!r}).get_all('Provides-Extra') or [])",
        "missing = sorted(set(requested) - provided)",
        "if missing:",
        "    sys.exit('extras not defined by the package on branch %s: %s' % (branch, ', '.join(missing)))",
    ])
    node.exec_in(args.ctid, f"{base}/venv/bin/python - <<'PY'\n{check}\nPY\n")
    print(f"  repo cloned into venv; package imports; extras verified: {', '.join(requested) or '(none)'}.")


def install_oracles(node: Node, args, name: str) -> None:
    """Run the repo's own scripts/install_oracles.sh inside the container -- the single place
    every oracle build recipe lives -- pointed at the container venv with SageMath enabled, then
    prove every oracle is discoverable so a broken install fails here."""
    if args.skip_oracles:
        print("[8/10] Skipping comparison-oracle install (--skip-oracles).")
        return
    print("[8/10] Installing comparison oracles (all engines + SageMath) via the in-repo script...")
    base = f"/opt/{name}"
    script = f"{base}/src/scripts/install_oracles.sh"
    debug_env = "DEBUG=1 " if node.verbose else ""
    node.exec_in(args.ctid,
                 f"test -f {shlex.quote(script)} || {{ echo 'oracle installer missing at {script}' >&2; exit 1; }}\n"
                 f"{debug_env}PIP={shlex.quote(base + '/venv/bin/pip')} "
                 "PIP_INSTALL_FLAGS='' "
                 f"PY={shlex.quote(base + '/venv/bin/python')} "
                 "INSTALL_SAGE=1 ORACLE_HOME=/opt/oracles BIN_DIR=/usr/local/bin "
                 f"bash {shlex.quote(script)}\n",
                 stream=True)
    node.exec_in(args.ctid,
                 f"py={shlex.quote(base + '/venv/bin/python')}\n"
                 "fail=0\n"
                 "for mod in knot_floer_homology snappy regina khoca; do\n"
                 '  if "$py" -c "import $mod" >/dev/null 2>&1; then echo "   $mod: import OK";\n'
                 '  else echo "   $mod: IMPORT FAILED" >&2; fail=1; fi\n'
                 "done\n"
                 "for exe in knotjob javakh khtpp kk khoho sage; do\n"
                 '  if command -v "$exe" >/dev/null 2>&1; then echo "   $exe: on PATH";\n'
                 '  else echo "   $exe: NOT ON PATH" >&2; fail=1; fi\n'
                 "done\n"
                 '[ "$fail" -eq 0 ] || { echo "oracle verification FAILED" >&2; exit 1; }\n')
    print("  verified: all oracles discoverable (4 imports + 6 on PATH, incl. sage).")


def setup_mdns(node: Node, ctid: int, mdns_name: str) -> None:
    print(f"[9/10] Advertising {mdns_name}.local over mDNS (Avahi)...")
    node.exec_in(ctid,
                 f"sed -ri 's/^#?host-name=.*/host-name={mdns_name}/' /etc/avahi/avahi-daemon.conf\n"
                 "systemctl enable --now avahi-daemon\n"
                 "systemctl restart avahi-daemon\n"
                 "systemctl is-active --quiet avahi-daemon\n"
                 "avahi-daemon --check\n"
                 f"grep -x 'host-name={mdns_name}' /etc/avahi/avahi-daemon.conf\n")
    print(f"  advertising {mdns_name}.local via Avahi.")


def smoke_test(node: Node, args, login: Login) -> None:
    if not args.smoke:
        print("[10/10] No --smoke command given; skipping smoke test.")
        return
    print(f"[10/10] Smoke test as {login.user!r}: {args.smoke}")
    node.exec_in(args.ctid,
                 f"cd {login.repo} && runuser -u {shlex.quote(login.user)} -- {login.python} {args.smoke}",
                 stream=True)
    print("  smoke test passed.")


def report(login: Login) -> None:
    host = login.address or login.mdns_name + ".local"
    bench = f"{login.repo}/scripts/bench_grid_floer.py"
    print("\nDone.")
    print(f"  Container {login.ctid} is up{(' at ' + login.address) if login.address else ''} on {login.node}.")
    print(f"  Repo at {login.repo}, venv python at {login.python}.")
    print(f"  Advertised over mDNS as {login.mdns_name}.local (link-local; an mDNS reflector is "
          f"needed to cross VLANs).")
    print(f"\n  Connect with the key:  ssh -i {login.key_path} {login.user}@{host}")
    print(f"  Or run one command:    python tools/ct_exec.py --ctid {login.ctid} -- nproc")
    print("  Then, e.g. the scaling sweep (synthetic sizes push the generator count; the scheduler"
          " uses the container's cores):")
    print(f"    {login.python} {bench} \\")
    print("        --sizes 8 9 10")
    print("  Tighten the RAM ceiling to study spilling/feasibility:  add  --mem-cap-gib <GiB>")
    print("  Or the full validation sweep (--heavy is the only opt-in):")
    print(f"        cd {login.repo} && {login.python} -m pytest --heavy -v")
    print(f"\n  Update the code later:  git -C {login.repo} pull")


def provision(args, client, save_login=write_login_file) -> Login:
    """Build the container over an open paramiko connection to the node and return its login.
    `save_login(login)` records the generated login as soon as the container accepts it and
    returns where it put it; the default writes the login file beside this script."""
    # Streamed build output carries non-ASCII that a Windows console codepage cannot encode.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    check_arguments(args)
    name = project_name(args.repo)
    node = Node(client, args.host, args.debug)
    preflight(node)
    cpus = plan_cpus(node, args)
    ensure_template(node, args.template, args.template_storage)
    create_container(node, args, cpus)
    wait_for_network(node, args.ctid)
    set_install_service_policy(node, args.ctid)   # brackets every apt install below
    install_base_packages(node, args)
    login = setup_login(node, args, name, save_login)
    install_repo(node, args, name)
    install_oracles(node, args, name)
    node.exec_in(args.ctid, f"chown -R {shlex.quote(login.user)}: /opt/{name}\n")
    setup_mdns(node, args.ctid, login.mdns_name)
    clear_install_service_policy(node, args.ctid)
    smoke_test(node, args, login)
    report(login)
    return login


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Host environment -- parameterized, no cluster assumptions:
    parser.add_argument("--host", required=True,
                        help="login on the Proxmox node, user@host[:port]: root, or an account "
                        "with passwordless sudo")
    parser.add_argument("--cores", type=int, required=True,
                        help="logical CPUs for the container, given as whole physical cores split "
                        "evenly over the node's NUMA nodes (so a multiple of NUMA nodes x threads "
                        "per core)")
    parser.add_argument("--memory", type=int, required=True,
                        help="RAM in MiB; the heavy tier (grid number 10) wants 32768 or more")
    parser.add_argument("--rootfs-storage", default="local-lvm",
                        help="Proxmox storage pool for the container rootfs (default local-lvm)")
    parser.add_argument("--ctid", type=int, default=250, help="container ID (default 250)")
    parser.add_argument("--hostname", default="tetradrome", help="container hostname")
    parser.add_argument("--swap", type=int, default=512, help="swap in MiB (default 512)")
    parser.add_argument("--rootfs-size", type=int, default=32,
                        help="rootfs size in GiB (default 32; the oracle suite -- SageMath plus "
                        "the built gold-masters -- needs the room)")
    parser.add_argument("--template-storage", default="local", help="template storage")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--bridge", default="vmbr0", help="network bridge (default vmbr0)")
    parser.add_argument("--ip", default="dhcp",
                        help="container IPv4: 'dhcp' or CIDR like 10.0.0.5/24 (default dhcp)")
    parser.add_argument("--gateway", default="", help="default gateway (required with a static --ip)")
    parser.add_argument("--vlan", type=int, default=0, help="VLAN tag for the NIC (0 = untagged)")
    parser.add_argument("--hwaddr", default="",
                        help="MAC address for the NIC, such as the one a DHCP reservation is keyed on, so a "
                             "rebuilt container keeps its address (default: Proxmox picks one)")
    parser.add_argument("--nameserver", default="", help="DNS server(s) (default: from DHCP)")
    parser.add_argument("--searchdomain", default="", help="DNS search domain (default: from DHCP)")
    parser.add_argument("--tags", default="tetradrome;compute",
                        help="Proxmox tags (semicolon-separated)")
    parser.add_argument("--ssh-user", default="tetradrome",
                        help="container login created with sshd, password and key auth (default tetradrome)")
    parser.add_argument("--mdns-hostname", default="",
                        help="hostname advertised over mDNS as <name>.local "
                        "(default: the container --hostname)")
    parser.add_argument("--recreate", action="store_true",
                        help="destroy and rebuild if the CTID already exists")
    # Project -- baked in, overridable:
    parser.add_argument("--repo", default=DEFAULT_REPO, help="git URL to clone")
    parser.add_argument("--branch", default="main", help="repo branch (default main)")
    parser.add_argument("--extras", default="all",
                        help="pip extras to install (default 'all': every CPU tier + KnotInfo + "
                        "pytest, so the box runs the full suite with no opt-in; GPU cupy stays "
                        "box-specific)")
    parser.add_argument("--smoke", default=DEFAULT_SMOKE,
                        help="command (venv python, in the repo dir, as the login) to smoke-test; '' to skip")
    parser.add_argument("--apt", default="", help="extra apt packages beyond the base")
    parser.add_argument("--skip-oracles", action="store_true",
                        help="skip installing the comparison oracles (SageMath, KnotJob, JavaKh, "
                        "kht++, knotkit, KhoHo, and the pip oracles regina/khoca/snappy); the box "
                        "still runs the native suite")
    parser.add_argument("--debug", "-v", action="store_true",
                        help="verbose: echo every remote command and stream all output live "
                        "(default: quiet, showing step progress, warnings, and full output only "
                        "on failure)")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        client = connect(args.host)
        try:
            provision(args, client)
        finally:
            client.close()
    except ProvisionError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
