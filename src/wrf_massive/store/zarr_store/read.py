# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading a store or a partial copy of it (made by `store pull`), and the HEALPix index helpers."""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import xarray as xr

from wrf_massive.store.zarr_store import layout as L

MANIFEST = ".wrf-massive-pull.json"  # in the root of a copy made by `store pull`


def read_manifest(path: pathlib.Path) -> dict | None:
    """What a partial copy holds, `{domain: {"vars": [...] | None, "sims": [...] | None}}` (None: all), or None for
    a complete store."""
    f = pathlib.Path(path) / MANIFEST
    return json.loads(f.read_text()) if f.exists() else None


def open_domain(path: str | pathlib.Path, domain: str, sort: bool = True, **open_kwargs) -> xr.Dataset:
    """A domain of a store as lazy xarray dataset, sorted by `sim_begin` (stable) unless `sort=False`.

    For a partial copy, only the sims and variables it holds are in the dataset, so asking for anything else raises
    instead of returning NaN. Only frames with `written=True` hold data. `open_kwargs` go to `xr.open_zarr`, e.g.
    `chunks=None` for lazy arrays without dask (each access reads through zarr directly).
    """
    path = pathlib.Path(path)
    if not (path / domain / "zarr.json").exists():
        raise KeyError(f"Domain '{domain}' is not in {path}" + _copy_hint(path))
    with warnings.catch_warnings():  # zarr: consolidated metadata is not in the v3 spec yet
        warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
        # The group dir itself: there is no root consolidated metadata
        ds = xr.open_zarr(path / domain, **({"consolidated": True} | open_kwargs))

    manifest = read_manifest(path)
    if manifest is not None:
        if domain not in manifest:
            raise KeyError(f"Domain '{domain}' was not pulled into this copy ({path}).")
        held = manifest[domain]
        if held.get("sims") is not None:
            ds = ds.isel(sim=np.flatnonzero(np.isin(ds["sim_name"].values, held["sims"])))
        if held.get("vars") is not None:
            ds = ds.drop_vars([v for v in data_vars(ds) if v not in held["vars"]])
    if sort:
        ds = ds.isel(sim=np.argsort(ds["sim_begin"].values, kind="stable"))
    return ds


def data_vars(ds: xr.Dataset) -> list[str]:
    """Time-dependent data variables of an opened domain."""
    return [v for v in ds.data_vars if ds[v].dims[:2] == ("sim", "lead_time") and v not in L.NON_DATA]


def hp_index(ds: xr.Dataset, order: int) -> xr.DataArray:
    """HEALPix (nested) cell of every grid point at `order` <= 23, from `hp_ipix` by bit shift (exact)."""
    if not 0 <= order <= L.HP_ORDER:
        raise ValueError(f"order must be within 0..{L.HP_ORDER}, got {order}.")
    ipix = ds["hp_ipix"].astype("uint64")
    return (ipix >> np.uint64(2 * (L.HP_ORDER - order))).rename(f"hp_ipix_{order}")


def select_cells(ds: xr.Dataset, cells, order: int) -> xr.DataArray:
    """Boolean (y, x) mask of the grid points inside the HEALPix cells `cells` at `order`."""
    return xr.DataArray(np.isin(hp_index(ds, order).values, np.asarray(cells, dtype="uint64")), dims=L.STATIC_DIMS)


def _copy_hint(path: pathlib.Path) -> str:
    manifest = read_manifest(path)
    return f" (partial copy holding {sorted(manifest)})" if manifest else ""
