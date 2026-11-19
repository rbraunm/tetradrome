# SPDX-License-Identifier: Apache-2.0

"""The grid each knot's Floer invariants are computed on, built once and kept.

A knot's grid is built from its reduced crossing diagram (``grid_construction``) and shrunk by
grid moves (``grid_moves``); for a non-alternating knot the shrinking search is the slow part.
The result is kept in one JSON file per reduced diagram, in ``grids/`` under the cache directory
the benchmark figures use (``TETRADROME_CACHE_DIR`` moves both). A file records the grid, the
diagram it was built from, when it was built, and the generator stamp: a digest of the code that
reduces diagrams, builds grids and moves them. An entry whose stamp differs from the running
code's is stale and is built again; nothing else expires it.

How a grid is found: the reduced diagram's crossing count plus two is the knot's smallest grid
when the diagram is alternating, so that grid is used as built. Otherwise a grid is built from
every starting arc of the diagram in both directions, each is shrunk by destabilizations reached
through at most ``SEARCH_DEPTH`` commutations at a time, and the smallest is kept, the first
in canonical order on a tie.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ...diagrams.model import PDCode
from ...diagrams.reduction import reduced, relabellings
from ...scheduler.calibration_store import store_directory
from .grid import GridDiagram
from .grid_construction import built
from .grid_moves import simplified

logger = logging.getLogger(__name__)

_FORMAT = 1
SEARCH_DEPTH = 6
_GENERATOR = ("diagrams/reduction.py", "diagrams/leveling.py", "engines/floer/grid.py",
              "engines/floer/grid_construction.py", "engines/floer/grid_moves.py",
              "engines/floer/grid_store.py")


@dataclass(frozen=True)
class GridSource:
    """Where a Floer result's grid came from. ``diagram`` is the reduced crossing diagram it
    was built from; ``source`` is "built" when this call built it and "stored" when it was read
    from ``stored_in``; ``built_at`` is when it was built (UTC); ``generator`` is the stamp of
    the code that built it."""

    o_markers: tuple
    x_markers: tuple
    diagram: PDCode
    source: str
    built_at: str
    generator: str
    stored_in: str

    @property
    def grid(self) -> GridDiagram:
        return GridDiagram(list(self.o_markers), list(self.x_markers))

    @property
    def size(self) -> int:
        return len(self.o_markers)


@functools.lru_cache(maxsize=1)
def generator_stamp() -> str:
    """A digest of the code that turns a crossing diagram into a grid."""
    package = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in _GENERATOR:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((package / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _alternating(diagram: PDCode) -> bool:
    """Whether the knot passes under and over in turn. In KnotInfo's labelling the arc labelled
    k enters one crossing; it enters under when it is that crossing's first label."""
    entered_under = {}
    for crossing in diagram:
        a, b, _, d = crossing
        entered_under[a] = True
        entered_under[d if b == d % (2 * len(diagram)) + 1 else b] = False
    labels = sorted(entered_under)
    return all(entered_under[label] != entered_under[labels[(k + 1) % len(labels)]]
               for k, label in enumerate(labels))


def smallest_grid(diagram: PDCode) -> GridDiagram:
    """The grid this module keeps for a reduced diagram, found as the module describes."""
    if not diagram or _alternating(diagram):
        return built(diagram)
    best = None
    for code in relabellings(diagram):
        candidate = simplified(built(code), SEARCH_DEPTH)
        if best is None or candidate.n < best.n:
            best = candidate
    return best


def _entry_path(diagram: PDCode, directory: Path) -> Path:
    key = hashlib.sha256(json.dumps([list(crossing) for crossing in diagram]).encode())
    return directory / f"grid-{key.hexdigest()[:32]}.json"


def _read(path: Path, diagram: PDCode) -> tuple[dict | None, str]:
    if not path.exists():
        return None, "no grid is stored for this diagram"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["format"] != _FORMAT:
            return None, "the stored grid is in an older format"
        if [tuple(crossing) for crossing in stored["diagram"]] != list(diagram):
            raise ValueError("it holds another diagram's grid")
        if stored["generator"] != generator_stamp():
            return None, "the code that builds grids changed since this one was built"
        return stored, ""
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"unreadable stored grid {path}: {error}; delete it") from error


def knot_grid(pd: PDCode, directory: Path | None = None) -> GridSource:
    """The grid for the knot with PD code ``pd``: read from the store when it holds a grid of
    this diagram from the running code, otherwise built and stored."""
    diagram = reduced(pd)
    directory = (directory or store_directory()) / "grids"
    path = _entry_path(diagram, directory)
    stored, why = _read(path, diagram)
    if stored is not None:
        return GridSource(tuple(stored["o"]), tuple(stored["x"]), diagram, "stored",
                          stored["built_at"], stored["generator"], str(path))
    logger.info("building the grid of a %d-crossing reduced diagram: %s", len(diagram), why)
    grid = smallest_grid(diagram)
    built_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    record = {"format": _FORMAT, "generator": generator_stamp(), "built_at": built_at,
              "diagram": [list(crossing) for crossing in diagram], "o": grid.O, "x": grid.X}
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temporary, path)
    return GridSource(tuple(grid.O), tuple(grid.X), diagram, "built", built_at,
                      generator_stamp(), str(path))
