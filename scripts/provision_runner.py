#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Provision a lightweight Proxmox LXC to run the Tetradrome grid scaling sweep.

Run this from anywhere; it drives a Proxmox node over SSH (--host root@<node>, required) and
never runs against a local node. SSH is spoken directly by a pure-Python client (paramiko),
not by shelling out to ssh(1): the node password is prompted once via getpass and held only
in memory for the auth handshake -- it never reaches a command line, file, or log. One
authenticated connection is opened and reused for every command.

It creates an unprivileged Debian LXC (pure compute -- no Docker, nesting, or GPU), installs
Tetradrome into a venv with the full runnable suite by default (every CPU acceleration tier,
the KnotInfo backend, and pytest -- so the box runs the suite, pytest --heavy and all, with no
opt-in beyond the test flags), enables sshd with a dedicated login (password + key auth) so the
prepared box can be used directly over SSH, advertises its hostname as <name>.local over mDNS
(Avahi) so it is reachable without a DNS record, smoke-tests a tiny sweep, and prints how to run
the full sweep at the container's core count with NUMA pinning.

The login is --ssh-user (default 'tetradrome') and owns the install. Each provision generates,
fresh, both a password and an ed25519 keypair: the password and the private key are written to
chmod-600 files beside this script (gitignored), and the public key is installed into the login's
authorized_keys, so the box accepts both password and key auth. The private key is generated on
this controller and never leaves it -- only the public line is pushed to the container -- and
`ssh -i <ctNNN-ssh-key>` (or paramiko with that key) drives the box non-interactively. These two
generated *container* credentials are what the tool persists, deliberately; the prompted *node*
password is still held only in memory and never reaches a command line, file, or log.

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
import secrets
import shlex
import sys
from collections import namedtuple

try:
    import paramiko
    # cryptography ships as a hard paramiko dependency, so it is always present here.
    # paramiko 5.x has no ed25519 generator, so the keypair is built with cryptography.
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


def exec_in_stdin(ctid: int, cmd: str, stdin_data: str) -> None:
    """Run an in-container command, feeding stdin_data to it over the channel's stdin so a
    secret never lands on a command line or in the printed log (same care as the node
    password). Fail loud on nonzero exit."""
    full = f"pct exec {ctid} -- {cmd}"
    print(f"$ {full}  (stdin withheld)")
    chan = _CLIENT.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(full)
    chan.sendall(stdin_data.encode())
    chan.shutdown_write()
    out = chan.makefile("r")
    for line in iter(out.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = chan.recv_exit_status()
    if rc != 0:
        sys.exit(f"command failed (exit {rc}) on {SSH_TARGET}: {full}")


def set_install_service_policy(ctid: int) -> None:
    """Install a policy-rc.d that returns 101 so apt maintainer scripts do not (re)start
    services during package install. deb-systemd-invoke honors this (it execs the helper
    before any systemctl call and skips on 101), which is what otherwise prints the
    'disabled or static unit' note and the 'Could not execute systemctl' die on an openssh
    upgrade. We start every service we need explicitly afterward, with verification, and a
    direct systemctl call ignores policy-rc.d -- so this removes a half-managed path we do
    not want, never a working one."""
    print("[*] Suppressing apt service auto-start (policy-rc.d 101); services started explicitly below.")
    exec_in(ctid,
            "printf '#!/bin/sh\\nexit 101\\n' > /usr/sbin/policy-rc.d\n"
            "chmod 0755 /usr/sbin/policy-rc.d\n"
            "test -x /usr/sbin/policy-rc.d\n")


def clear_install_service_policy(ctid: int) -> None:
    """Remove the install-time policy-rc.d so normal service management resumes on the box."""
    exec_in(ctid, "rm -f /usr/sbin/policy-rc.d\n")


def generate_keypair(comment: str) -> tuple[str, str]:
    """Generate a fresh ed25519 keypair on this controller. Returns (OpenSSH private-key
    text, authorized_keys public line). The private key is created here and never leaves
    this machine -- only the public line is pushed to the container."""
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


def install_authorized_key(ctid: int, args, public_line: str) -> None:
    """Install the provisioning public key into the login's authorized_keys (key auth, in
    addition to the password). The public line is not secret; its postcondition is asserted."""
    print(f"  installing the provisioning SSH key for {args.ssh_user!r} (key auth)...")
    user = shlex.quote(args.ssh_user)
    key_q = shlex.quote(public_line)
    exec_in(ctid,
            f"user={user}\n"
            'home=$(getent passwd "$user" | cut -d: -f6)\n'
            '[ -n "$home" ] || { echo "no home dir for $user" >&2; exit 1; }\n'
            'install -d -m 700 -o "$user" -g "$user" "$home/.ssh"\n'
            f'printf "%s\\n" {key_q} > "$home/.ssh/authorized_keys"\n'
            'chmod 600 "$home/.ssh/authorized_keys"\n'
            'chown "$user":"$user" "$home/.ssh/authorized_keys"\n'
            # postcondition: the key really landed (grep -F exits nonzero -> set -e fails loud)
            f'grep -qF {key_q} "$home/.ssh/authorized_keys"\n'
            'echo "verified: authorized_keys installed for $user"\n')


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
    print(f"[1/8] Ensuring template {template} is cached on {template_storage}...")
    if template in capture(f"pveam list {template_storage}").stdout:
        print("  already cached.")
        return
    run("pveam update")
    run(f"pveam download {template_storage} {template}")


def create_container(args) -> None:
    print(f"[2/8] Creating LXC {args.ctid} ({args.hostname})...")
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
    print("[3/8] Waiting for container network...")
    exec_in(ctid, 'for i in $(seq 1 30); do '
                  'getent hosts github.com >/dev/null 2>&1 && exit 0; sleep 2; done; '
                  'echo "no network in container after 60s" >&2; exit 1')


def install_repo(ctid: int, args, name: str) -> None:
    extras = f"[{args.extras}]" if args.extras else ""
    print(f"[4/8] Installing {args.repo} ({args.branch}"
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
    verify_install(ctid, args, name)


def verify_install(ctid: int, args, name: str) -> None:
    """Assert step 4's postcondition. pip exits 0 even when a requested extra does not
    exist -- it prints a WARNING and installs the bare package -- which is a silent
    fallback this script must not inherit (a missing extra once surfaced only via the
    optional smoke test, three steps later). So prove the package imports and that the
    installed metadata defines every requested extra; pip exiting 0 with the extra
    present implies its dependencies resolved and installed."""
    print("  verifying the install (package import + requested extras defined)...")
    base = f"/opt/{name}"
    requested = [e.strip() for e in args.extras.split(",") if e.strip()]
    py = "\n".join([
        "import importlib.metadata as md, sys",
        f"import {name}" if name.isidentifier() else "",
        f"requested = {requested!r}",
        f"branch = {args.branch!r}",
        f"provided = set(md.metadata({name!r}).get_all('Provides-Extra') or [])",
        "missing = sorted(set(requested) - provided)",
        "if missing:",
        "    sys.exit('FAIL: extras not defined by the package on branch %s: %s. '",
        "             'pip only warns on an unknown extra and installs the bare package, '",
        "             'so this install is incomplete.' % (branch, ', '.join(missing)))",
        "print('install verified: package imports; extras defined:', ', '.join(requested) or '(none requested)')",
    ])
    exec_in(ctid, f"{base}/venv/bin/python - <<'PY'\n{py}\nPY\n")


def setup_ssh(ctid: int, args, name: str, password: str, public_line: str) -> None:
    print(f"[5/8] Enabling sshd and the {args.ssh_user!r} login on the container...")
    base = f"/opt/{name}"
    user = shlex.quote(args.ssh_user)
    exec_in(ctid,
            "export DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8\n"
            "apt-get install -y -qq openssh-server\n"
            f"id -u {user} >/dev/null 2>&1 || useradd --create-home --shell /bin/bash {user}\n"
            f"chown -R {user}:{user} {base}\n"   # the login owns its tools: run, pull, write caches
            "install -d -m 0755 /etc/ssh/sshd_config.d\n"
            "printf 'PasswordAuthentication yes\\nPubkeyAuthentication yes\\n'"
            " > /etc/ssh/sshd_config.d/10-tetradrome.conf\n"
            "systemctl enable --now ssh\n"
            "systemctl restart ssh\n"
            # assert the postcondition: the unit is active and password auth is effective
            # in the RUNNING config (apt postinst noise like deb-systemd-invoke errors is
            # irrelevant once these pass). grep without -q reads all input -- no SIGPIPE
            # under pipefail -- and echoes the matched line as proof in the log.
            "systemctl is-active --quiet ssh\n"
            "sshd -T 2>/dev/null | grep -i '^passwordauthentication yes'\n"
            "echo 'verified: sshd is active with password auth effective'\n")
    # set the password over stdin so it never appears on a command line or in the log
    exec_in_stdin(ctid, "chpasswd", f"{args.ssh_user}:{password}\n")
    install_authorized_key(ctid, args, public_line)


def setup_mdns(ctid: int, mdns_name: str) -> None:
    print(f"[6/8] Advertising {mdns_name}.local over mDNS (Avahi)...")
    exec_in(ctid,
            "export DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8\n"
            "apt-get install -y -qq avahi-daemon libnss-mdns\n"
            # advertise a deterministic host-name (defaults to the container hostname)
            f"sed -ri 's/^#?host-name=.*/host-name={mdns_name}/' /etc/avahi/avahi-daemon.conf\n"
            "systemctl enable --now avahi-daemon\n"
            "systemctl restart avahi-daemon\n"
            # assert the postcondition rather than trusting the restart's exit alone
            "systemctl is-active --quiet avahi-daemon\n"
            "avahi-daemon --check\n"
            f"grep -x 'host-name={mdns_name}' /etc/avahi/avahi-daemon.conf\n"
            "echo 'verified: avahi-daemon is active and advertising the host-name above'\n")


def smoke_test(ctid: int, args, name: str) -> None:
    if not args.smoke:
        print("[7/8] No --smoke command given; skipping smoke test.")
        return
    print(f"[7/8] Smoke test: {args.smoke}")
    base = f"/opt/{name}"
    exec_in(ctid, f"cd {base}/src && {base}/venv/bin/python {args.smoke}")


def container_ip(ctid: int) -> str:
    parts = capture(f"pct exec {ctid} -- hostname -I").stdout.strip().split()
    return parts[0] if parts else ""


def write_private_key(ctid: int, private_text: str, public_line: str) -> str:
    """Persist the generated keypair beside this script (gitignored): the private key at
    ctNNN-ssh-key (chmod 600 from creation, no world-readable window) and its public line
    at ctNNN-ssh-key.pub. Returns the private-key path -- what a client passes to ssh -i."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    priv_path = os.path.join(script_dir, f"ct{ctid}-ssh-key")
    pub_path = priv_path + ".pub"
    # create the private key already-restricted rather than chmod-ing an open file afterward
    fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(private_text)
    os.chmod(priv_path, 0o600)
    with open(pub_path, "w") as f:
        f.write(public_line + "\n")
    return priv_path


def write_credentials(args, name: str, password: str, ip: str, mdns_name: str,
                      key_path: str) -> str:
    """Write the generated login to a chmod-600 file beside this script (gitignored). The
    password and the private key are the credentials the tool persists; the prompted node
    password is never written."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, f"ct{args.ctid}-ssh-credentials.txt")
    base = f"/opt/{name}"
    lines = [
        "# provision_runner.py generated container login -- DO NOT COMMIT (gitignored).",
        f"host:     {args.hostname} (CT {args.ctid}) on {SSH_TARGET}",
        f"address:  {ip or '(unknown -- run: pct exec %d -- hostname -I)' % args.ctid}",
        f"mdns:     {mdns_name}.local",
        f"user:     {args.ssh_user}",
        f"password: {password}",
        f"ssh-key:  {key_path}   (ssh -i this; key auth, in addition to the password)",
        f"repo:     {base}/src",
        f"venv:     {base}/venv/bin/python",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o600)
    return path


def report(ctid: int, args, name: str, ip: str, creds_path: str, key_path: str,
           mdns_name: str) -> None:
    base = f"/opt/{name}"
    python = f"{base}/venv/bin/python"
    bench = f"{base}/src/scripts/bench_grid_floer.py"
    host = ip or mdns_name + ".local"
    print("\n[8/8] Done.")
    print(f"  Container {ctid} is up{(' at ' + ip) if ip else ''} on {SSH_TARGET}.")
    print(f"  Repo at {base}/src, venv python at {python}.")
    print(f"  SSH login '{args.ssh_user}' enabled (password + key auth); credentials in:")
    print(f"    {creds_path}   (chmod 600, gitignored)")
    print(f"  Provisioning key (private, chmod 600, gitignored):")
    print(f"    {key_path}   (and {key_path}.pub)")
    print(f"  Advertised over mDNS as {mdns_name}.local (link-local; an mDNS reflector is needed to cross VLANs).")
    print(f"\n  Connect with the key:  ssh -i {key_path} {args.ssh_user}@{host}")
    print(f"  Or with the password:  ssh {args.ssh_user}@{ip or '<container-ip>'}   (or  ssh {args.ssh_user}@{mdns_name}.local)")
    print("  Then, e.g. the scaling sweep (synthetic sizes push the generator count; the scheduler"
          " uses the container's cores):")
    print(f"    {python} {bench} \\")
    print("        --sizes 8 9 10 11")
    print("  Tighten the RAM ceiling to study spilling/feasibility:  add  --mem-cap-gib <GiB>")
    print("  Or the full validation sweep (--heavy is the only opt-in):")
    print(f"        cd {base}/src && {python} -m pytest --heavy -v")
    print(f"\n  Update the code later:  git -C {base}/src pull")


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
    parser.add_argument("--ssh-user", default="tetradrome",
                        help="container login created with sshd + password auth (default tetradrome)")
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
        set_install_service_policy(args.ctid)   # brackets every apt install below
        install_repo(args.ctid, args, name)
        password = secrets.token_urlsafe(18)
        private_text, public_line = generate_keypair(f"tetradrome-provisioner-ct{args.ctid}")
        setup_ssh(args.ctid, args, name, password, public_line)
        mdns_name = args.mdns_hostname or args.hostname
        setup_mdns(args.ctid, mdns_name)
        clear_install_service_policy(args.ctid)  # apt installs done; restore normal service mgmt
        ip = container_ip(args.ctid)
        key_path = write_private_key(args.ctid, private_text, public_line)
        creds_path = write_credentials(args, name, password, ip, mdns_name, key_path)
        smoke_test(args.ctid, args, name)
        report(args.ctid, args, name, ip, creds_path, key_path, mdns_name)
    finally:
        _CLIENT.close()


if __name__ == "__main__":
    main()
