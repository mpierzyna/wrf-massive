import matplotlib

matplotlib.use("Agg")  # headless, no display needed

import numpy as np
import xarray as xr
from click.testing import CliRunner

from wrf_massive.plotting import cli, plot_domain


def _make_geo_em(path):
    """Write a minimal geo_em-like file (terrain + lat/lon on the mass grid)."""
    sn, we = 5, 7
    lat, lon = np.meshgrid(np.linspace(28, 29, sn), np.linspace(-17, -15, we), indexing="ij")
    hgt = np.linspace(0, 1000, sn * we).reshape(sn, we)
    ds = xr.Dataset(
        {
            "HGT_M": (("Time", "south_north", "west_east"), hgt[None]),
            "XLAT_M": (("Time", "south_north", "west_east"), lat[None]),
            "XLONG_M": (("Time", "south_north", "west_east"), lon[None]),
        },
        attrs={"DX": 2000.0},
    )
    ds.to_netcdf(path)


def test_plot_domain_returns_axes(tmp_path):
    geo = tmp_path / "geo_em.d01.nc"
    _make_geo_em(geo)
    ax = plot_domain(geo, add_coastlines=False)
    assert ax.collections  # pcolormesh drew a QuadMesh
    assert "geo_em.d01.nc" in ax.get_title()
    assert "2 km" in ax.get_title()  # DX=2000 m -> 2 km, 7x5 grid


def test_plot_domain_savefig(tmp_path):
    geo = tmp_path / "geo_em.d01.nc"
    _make_geo_em(geo)
    out = tmp_path / "domain.png"
    plot_domain(geo, add_coastlines=False, savefig=out)
    assert out.exists() and out.stat().st_size > 0


def test_cli_domain(tmp_path):
    geo = tmp_path / "geo_em.d01.nc"
    _make_geo_em(geo)
    result = CliRunner().invoke(cli, ["domain", str(geo), "--no-coastlines"])
    assert result.exit_code == 0, result.output
    # output is auto-derived next to the geo_em file as <stem>_domain.png
    out = tmp_path / "geo_em.d01_domain.png"
    assert out.exists() and out.stat().st_size > 0
