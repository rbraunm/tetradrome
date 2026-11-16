# SPDX-License-Identifier: Apache-2.0

"""The scheduler's measurements, kept between processes.

What a kind of job costs on a device, and what a placement costs to dispatch, belong to the code
and the machine, not to one run. They are measured once for this code on this machine and kept in
a JSON file, one per machine, under the user's cache directory (``TETRADROME_CACHE_DIR`` moves
it). A figure is measured again only when the code changes, which discards the whole file, or
when the file does not hold it yet.

The code's version is a hash of every source file in the package, so any edit counts. The
machine is its processor model and the CPUs and GPUs the scheduler may use; RAM ceilings are left
out because they bound what fits, not how fast it runs.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

from .inventory import Machine
from .routing import Calibration

logger = logging.getLogger(__name__)

_FORMAT = 1


def store_directory() -> Path:
    """Where the calibration files live: ``TETRADROME_CACHE_DIR`` when set, else the platform's
    per-user cache directory."""
    override = os.environ.get("TETRADROME_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA")
        if not root:
            raise RuntimeError("LOCALAPPDATA is not set; set TETRADROME_CACHE_DIR to choose where "
                               "the scheduler keeps its measurements")
        return Path(root) / "tetradrome"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "tetradrome"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tetradrome"


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """A hash of every Python source file in the installed package."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _processor() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def machine_identity(machine: Machine) -> dict:
    """What makes two machines run the same jobs at the same speed."""
    return {
        "architecture": platform.machine(),
        "processor": _processor(),
        "nodes": [[node.index, sorted(node.cores)] for node in machine.nodes],
        "gpus": [[gpu.index, gpu.vram_bytes] for gpu in machine.gpus],
    }


def store_path(machine: Machine, directory: Path) -> Path:
    identity = json.dumps(machine_identity(machine), sort_keys=True).encode()
    return directory / f"calibration-{hashlib.sha256(identity).hexdigest()[:16]}.json"


def _read(path: Path, code: str) -> dict | None:
    """The stored figures for ``code``, or None when there are none or they belong to other code.
    A file that cannot be read as a calibration raises, naming it."""
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["format"] != _FORMAT:
            raise ValueError(f"format {stored['format']}, expected {_FORMAT}")
        if stored["code"] != code:
            return None
        return stored["figures"]
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"unreadable scheduler calibration {path}: {error}; delete it and the "
                           f"scheduler measures again") from error


def load(calibration: Calibration, machine: Machine, directory: Path, code: str) -> None:
    """Give ``calibration`` this code's stored figures for this machine, if there are any."""
    path = store_path(machine, directory)
    figures = _read(path, code)
    if figures is None:
        logger.info("no stored scheduler calibration for this code on this machine; measuring")
        return
    calibration.fix(figures)
    logger.info("scheduler calibration loaded from %s", path)


def save(calibration: Calibration, machine: Machine, directory: Path, code: str) -> None:
    """Add the figures ``calibration`` has finished measuring to the stored ones. Stored figures
    are kept as they are; another process may have added some since this one loaded."""
    ready = calibration.ready()
    if not ready["rates"] and not ready["overheads"]:
        return
    path = store_path(machine, directory)
    figures = _read(path, code) or {"rates": [], "overheads": []}
    held_rates = {(entry["kind"], entry["device"]) for entry in figures["rates"]}
    held_overheads = {entry["placement"] for entry in figures["overheads"]}
    figures["rates"] += [entry for entry in ready["rates"]
                         if (entry["kind"], entry["device"]) not in held_rates]
    figures["overheads"] += [entry for entry in ready["overheads"]
                             if entry["placement"] not in held_overheads]
    directory.mkdir(parents=True, exist_ok=True)
    stored = {"format": _FORMAT, "code": code, "machine": machine_identity(machine),
              "figures": figures}
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(stored, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    calibration.fix(figures)
