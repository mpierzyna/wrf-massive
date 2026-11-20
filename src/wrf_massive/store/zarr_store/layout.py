# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Layout of the zarr store (names, dims, attrs) and helpers shared by init, writing and reading. See DESIGN.md."""

from __future__ import annotations

import contextlib
import fcntl
import json
import pathlib
from collections.abc import Iterator

import numpy as np
import zarr

SCHEMA_VERSION = 1

# Dims of the stored data arrays and the dims of a processed frame they come from
DIMS_3D = ("sim", "lead_time", "bottom_top", "y", "x")
DIMS_2D = ("sim", "lead_time", "y", "x")
FRAME_DIMS = {
    ("time", "bottom_top", "south_north", "west_east"): DIMS_3D,
    ("time", "south_north", "west_east"): DIMS_2D,
}
STATIC_DIMS = ("y", "x")
FRAME_STATIC_DIMS = ("south_north", "west_east")

COORDS_2D = ("XLAT", "XLONG")  # lat/lon of the mass grid: static coords of every domain
HP_ORDER = 23  # HEALPix order of `hp_ipix` (≈ 0.78 m cells)
WRITTEN = "written"
TIME_UNITS = "seconds since 1970-01-01 00:00:00"
CHUNK_KEYS = {"name": "default", "separator": "."}  # flat shard files, one directory per array (inodes)
SIM_CHUNK = 64  # chunk length along `sim` of the small per-sim arrays (written by init-domain only)

# Arrays of a domain group that are not data variables
NON_DATA = {"lead_time", "time", "sim_name", "sim_begin", "sim_settings", WRITTEN, "hp_ipix", *COORDS_2D}


def open_root(path: pathlib.Path, mode: str = "r") -> zarr.Group:
    return zarr.open_group(store=str(path), mode=mode, use_consolidated=False)


def open_domain(path: pathlib.Path, domain: str, mode: str = "r") -> zarr.Group:
    """The domain group. Its consolidated metadata is not used: arrays' own `zarr.json` are authoritative."""
    return zarr.open_group(store=str(path), path=domain, mode=mode, use_consolidated=False)


def data_vars(group: zarr.Group) -> list[str]:
    """Names of the time-dependent data arrays of a domain group (self-describing via their dimension names)."""
    return sorted(
        n for n, a in group.arrays() if n not in NON_DATA and tuple(a.metadata.dimension_names[:2]) == DIMS_2D[:2]
    )


def sim_index(group: zarr.Group, sim_name: str) -> int | None:
    """Position of sim `sim_name` on the `sim` dim, or None if it is not in the domain."""
    names = list(group["sim_name"][:])
    if names.count(sim_name) > 1:
        raise RuntimeError(f"sim '{sim_name}' is in the store {names.count(sim_name)} times.")
    return names.index(sim_name) if sim_name in names else None


def jsonable(attrs: dict) -> dict:
    """Attrs that can be stored as zarr (JSON) attributes; numpy scalars/arrays converted, the rest dropped."""
    out = {}
    for k, v in attrs.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        try:
            json.dumps(v, allow_nan=False)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


@contextlib.contextmanager
def exclusive_lock(path: pathlib.Path, what: str) -> Iterator[None]:
    """Exclusive POSIX lock on `path` (created if needed; works across nodes on GPFS), raises if already held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fd:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError(f"{what} is locked by another process ({path}).") from e
        try:
            yield
        finally:
            fcntl.lockf(fd, fcntl.LOCK_UN)


def domain_lock_path(path: pathlib.Path, domain: str) -> pathlib.Path:
    """Lock file of `init-domain` for one domain: next to the store, so copies of the store stay clean."""
    return path.parent / f"{path.name}.{domain}.lock"
