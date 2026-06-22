#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a command on a provision_runner.py container over SSH, using the generated key.

Reuses what scripts/provision_runner.py already wrote beside itself -- the private key
(scripts/ctNNN-ssh-key) and the login file (scripts/ctNNN-ssh-credentials.txt, for the
address and user) -- so there is no password and no per-command boilerplate. Point it at a
CTID and a command; the command's combined output streams live and this exits with the
remote command's status, so it composes and fails loud.

    python tools/ct_exec.py -- nproc
    python tools/ct_exec.py --ctid 250 -- numactl --hardware
    python tools/ct_exec.py -- "cd /opt/tetradrome/src && venv/bin/python -m pytest -q"
    python tools/ct_exec.py --host 10.0.0.5 --user tetradrome -- uptime

Put the command after `--` (so argparse does not eat its dashes) or pass it as one quoted
string. Address/user come from the credentials file unless --host/--user override them; the
key defaults to scripts/ctNNN-ssh-key unless --key overrides it.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import paramiko
except ImportError:
    sys.exit(
        "ct_exec needs paramiko (the pure-Python SSH client).\n"
        "Install it from an admin prompt: pip install 'tetradrome[provision]'  (or: pip install paramiko)"
    )

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def read_credentials(ctid: int) -> dict:
    """Parse the 'key: value' login file provision_runner wrote, or {} if it is absent."""
    path = os.path.join(SCRIPTS, f"ct{ctid}-ssh-credentials.txt")
    creds: dict = {}
    if not os.path.exists(path):
        return creds
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            creds[key.strip()] = value.strip()
    return creds


def resolve_target(args) -> tuple[str, str, str]:
    """Resolve (host, user, key_path) from --flags, falling back to the credentials file.
    Fails loud rather than guessing when a required value is missing or unusable."""
    creds = read_credentials(args.ctid)
    host = args.host or creds.get("address", "")
    user = args.user or creds.get("user", "")
    key_path = args.key or os.path.join(SCRIPTS, f"ct{args.ctid}-ssh-key")
    if not host or host.startswith("("):   # "(unknown -- run: ...)" sentinel from provisioning
        sys.exit(f"no usable address for CT {args.ctid} (creds say {host!r}); pass --host.")
    if not user:
        sys.exit(f"no user for CT {args.ctid}; pass --user.")
    if not os.path.exists(key_path):
        sys.exit(f"private key not found: {key_path} (run provision_runner.py, or pass --key).")
    return host, user, key_path


def exec_stream(client: "paramiko.SSHClient", command: str) -> int:
    """Run command, stream combined stdout+stderr live, return its exit status."""
    chan = client.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(command)
    stdout = chan.makefile("r")
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    return chan.recv_exit_status()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ctid", type=int, default=250, help="container ID (default 250)")
    parser.add_argument("--host", default="", help="override the address from the creds file")
    parser.add_argument("--user", default="", help="override the login from the creds file")
    parser.add_argument("--key", default="", help="override the private key path")
    parser.add_argument("--timeout", type=float, default=15.0, help="connect timeout seconds")
    parser.add_argument("command", nargs="*",
                        help="the command to run (put it after -- or quote it)")
    args = parser.parse_args()

    command = " ".join(args.command).strip()
    if not command:
        sys.exit("no command given. Example: python tools/ct_exec.py -- nproc")

    host, user, key_path = resolve_target(args)
    key = paramiko.Ed25519Key.from_private_key_file(key_path)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=user, pkey=key, look_for_keys=False,
                       allow_agent=False, timeout=args.timeout)
    except paramiko.AuthenticationException:
        sys.exit(f"key auth failed for {user}@{host} with {key_path}.")
    except (paramiko.SSHException, OSError) as e:
        sys.exit(f"cannot reach {user}@{host}: {e}")
    try:
        rc = exec_stream(client, command)
    finally:
        client.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
