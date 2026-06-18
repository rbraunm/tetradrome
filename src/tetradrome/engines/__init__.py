# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""Front-end engines: per-theory machinery that turns a diagram into the data an
invariant is read from. `cube` is the shared resolution-cube skeleton (the scaffold
the native Khovanov engine will build on)."""
from . import cube

__all__ = ["cube"]
