# SPDX-License-Identifier: Apache-2.0

"""Where the scheduler's benchmark figures are kept.

How fast a kind of job runs, what a placement costs to dispatch and what a fresh worker costs to
start belong to the machine and the software, not to one run. ``tetradrome.calibrate`` measures
them; this module keeps them: one JSON file per machine under the user's cache directory
(``TETRADROME_CACHE_DIR`` moves it).

A file is used only while both of its keys still hold:

- the machine: processor model, the CPUs the scheduler may use with their NUMA nodes and
  physical cores, and the GPUs. RAM, load, uptime and kernel are left out, so a reboot changes nothing.
- the software: every source file of the package, and the versions of the interpreter and the
  numerical libraries the kernels run on.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from .inventory import Machine

_FORMAT = 4
_LIBRARIES = ("numpy", "numba", "cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x")


def store_directory() -> Path:
    """Where the figures live: ``TETRADROME_CACHE_DIR`` when set, else the platform's per-user
    cache directory."""
    override = os.environ.get("TETRADROME_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA")
        if not root:
            raise RuntimeError("LOCALAPPDATA is not set; set TETRADROME_CACHE_DIR to choose where "
                               "the scheduler keeps its benchmark figures")
        return Path(root) / "tetradrome"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "tetradrome"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tetradrome"


def _installed(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def software_identity() -> dict:
    """What decides how fast this installation runs a job: the package's source and the versions
    of the interpreter and numerical libraries under it."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    libraries = {name: version for name in _LIBRARIES if (version := _installed(name))}
    return {"source": digest.hexdigest(), "python": platform.python_version(),
            "libraries": libraries}


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
        "nodes": [[node.index, [sorted(core) for core in node.physical_cores]]
                  for node in machine.nodes],
        "gpus": [[gpu.index, gpu.vram_bytes] for gpu in machine.gpus],
    }


def store_path(machine: Machine, directory: Path) -> Path:
    identity = json.dumps(machine_identity(machine), sort_keys=True).encode()
    return directory / f"calibration-{hashlib.sha256(identity).hexdigest()[:16]}.json"


def read(machine: Machine, directory: Path) -> tuple[dict | None, str]:
    """``(figures, "")`` when this machine's figures were measured with this software, else
    ``(None, why not)``. A file that cannot be read as a calibration raises, naming it."""
    path = store_path(machine, directory)
    if not path.exists():
        if any(directory.glob("calibration-*.json")):
            return None, "this machine's processors or GPUs are not the ones measured before"
        return None, "this machine has not been measured yet"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["format"] != _FORMAT:
            return None, "the stored figures are in an older format"
        if stored["software"] != software_identity():
            return None, "the code or its libraries changed since the last measurement"
        return stored["figures"], ""
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"unreadable scheduler calibration {path}: {error}; delete it, or run "
                           f"`python -m tetradrome.calibrate`") from error


def write(machine: Machine, directory: Path, figures: dict) -> Path:
    """Replace this machine's stored figures with ``figures``, measured with this software."""
    directory.mkdir(parents=True, exist_ok=True)
    path = store_path(machine, directory)
    stored = {"format": _FORMAT, "software": software_identity(),
              "machine": machine_identity(machine), "figures": figures}
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(stored, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path
