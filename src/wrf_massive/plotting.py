"""Small plotting helpers for WRF/WPS artifacts."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Tuple

import click
import xarray as xr

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_domain(
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


@click.group()
def cli():
    """Plotting utilities for WRF/WPS artifacts."""


@cli.command("domain")
@click.argument("geo_em_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--cmap", default="terrain", show_default=True, help="Matplotlib colormap for terrain height.")
@click.option("--coastlines/--no-coastlines", default=True, show_default=True, help="Draw coastlines.")
@click.option("--coastline-resolution", type=click.Choice(["10m", "50m", "110m"]), default="10m", show_default=True)
@click.option("--gridlines/--no-gridlines", default=True, show_default=True, help="Draw labelled lat/lon gridlines.")
def domain_cmd(geo_em_path, cmap, coastlines, coastline_resolution, gridlines):
    """Plot the terrain of a WPS domain from its GEO_EM_PATH (a geo_em.dNN.nc file)."""

    geo_em_path = pathlib.Path(geo_em_path)
    output = geo_em_path.with_name(geo_em_path.stem + "_domain.png")

    plot_domain(
        geo_em_path,
        cmap=cmap,
        add_coastlines=coastlines,
        coastline_resolution=coastline_resolution,
        add_gridlines=gridlines,
        savefig=output,
    )
    click.echo(f"-> Saved domain plot to {output}")


if __name__ == "__main__":
    cli()
