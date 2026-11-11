# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small plotting helpers for WRF/WPS artifacts."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Sequence, Tuple

import xarray as xr

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_domain_geo_em(
    geo_em_path: str | pathlib.Path,
    ax: "Axes | None" = None,
    cmap: str = "terrain",
    add_coastlines: bool = True,
    coastline_resolution: str = "10m",
    add_gridlines: bool = True,
    figsize: Tuple[float, float] = (8, 8),
    savefig: str | pathlib.Path | None = None,
) -> "Axes":
    """Plot the terrain of a WPS domain from its ``geo_em`` file on a cartopy map.

    Reads the static terrain height (``HGT_M``) and its lat/lon (``XLAT_M``/``XLONG_M``) from a
    ``geo_em.dNN.nc`` file (geogrid output) and draws it with pcolormesh plus coastlines. Uses the grid's
    own lon/lat on a PlateCarree transform, so it is agnostic to the domain's map projection.

    Parameters
    ----------
    geo_em_path : path to a ``geo_em.dNN.nc`` file.
    ax : existing cartopy ``GeoAxes`` to draw on; a new figure/axes is created if ``None``.
    cmap : matplotlib colormap for terrain height.
    add_coastlines : draw Natural Earth coastlines (needs the coastline dataset / network on first use).
    coastline_resolution : coastline resolution, one of ``"10m"``, ``"50m"``, ``"110m"``.
    add_gridlines : draw labelled lat/lon gridlines.
    figsize : figure size when creating a new axes.
    savefig : if given, save the figure to this path (tight bbox).

    Returns
    -------
    matplotlib.axes.Axes
        The (cartopy) axes drawn on; the figure is available as ``ax.figure``.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    geo_em_path = pathlib.Path(geo_em_path)
    with xr.open_dataset(geo_em_path) as ds:
        hgt = ds["HGT_M"].isel(Time=0).load()
        lat = ds["XLAT_M"].isel(Time=0).load()
        lon = ds["XLONG_M"].isel(Time=0).load()
        dx = ds.attrs.get("DX")

    proj = ccrs.PlateCarree()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    mesh = ax.pcolormesh(lon, lat, hgt, cmap=cmap, transform=proj, shading="auto")
    if add_coastlines:
        ax.coastlines(resolution=coastline_resolution, linewidth=0.8)
    ax.set_extent(
        [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        crs=proj,
    )
    if add_gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.top_labels = gl.right_labels = False

    cbar = ax.figure.colorbar(mesh, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Terrain height [m]")

    title = geo_em_path.name
    if dx is not None:
        title += f"  (dx = {dx / 1000:g} km, {hgt.shape[1]}x{hgt.shape[0]})"
    ax.set_title(title)

    if savefig is not None:
        ax.figure.savefig(savefig, bbox_inches="tight", dpi=150)

    return ax


def plot_domain_nml(namelist: str | pathlib.Path | None, args: Sequence[str] = (), save_file: str = "domains.png"):
    """Plot the domain layout of a WPS namelist without running geogrid.

    Runs `plot_domains.py` from WPS v4.7.0 (Anthony Islas, 2025; vendored in `wrf_massive/third_party/`) on
    `namelist`, saving to `save_file` when no interactive display is available. `args` are further command line
    arguments of the script, e.g. `["--info", "15"]`; they come last, so they override `namelist`/`save_file`
    (`-n`/`-s`). Without `namelist`, only `args` are passed (e.g. `["--help"]`). The script only has a `__main__`
    block, so it is run as a module with the arguments in `sys.argv`.
    """
    import runpy
    import sys

    defaults = [] if namelist is None else ["-n", str(namelist), "-s", str(save_file)]
    argv = sys.argv
    sys.argv = ["wrf-massive plot domain-nml", *defaults, *args]
    try:
        runpy.run_module("wrf_massive.third_party.plot_domains", run_name="__main__")
    finally:
        sys.argv = argv
