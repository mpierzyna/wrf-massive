# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless, no display needed

import numpy as np
import xarray as xr
from click.testing import CliRunner

from wrf_massive.cli import main
from wrf_massive.plotting import plot_domain_geo_em


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


def test_plot_domain_geo_em_returns_axes(tmp_path):
    geo = tmp_path / "geo_em.d01.nc"
    _make_geo_em(geo)
    ax = plot_domain_geo_em(geo, add_coastlines=False)
    assert ax.collections  # pcolormesh drew a QuadMesh
    assert "geo_em.d01.nc" in ax.get_title()
    assert "2 km" in ax.get_title()  # DX=2000 m -> 2 km, 7x5 grid


def test_cli_domain_geo_em(tmp_path):
    geo = tmp_path / "geo_em.d01.nc"
    _make_geo_em(geo)
    result = CliRunner().invoke(main, ["plot", "domain-geo-em", str(geo), "--no-coastlines"])
    assert result.exit_code == 0, result.output
    # output is auto-derived next to the geo_em file as <stem>_domain.png
    out = tmp_path / "geo_em.d01_domain.png"
    assert out.exists() and out.stat().st_size > 0


def test_cli_domain_nml(tmp_path, monkeypatch):
    """`plot domain-nml NAMELIST` runs the vendored WPS plot_domains.py, saving to domain.png by default and passing
    further options through."""
    import wrf_massive
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)  # no Natural Earth downloads
    monkeypatch.chdir(tmp_path)
    namelist = pathlib.Path(wrf_massive.__file__).parent / "templates" / "workspace" / "namelist.tmpl.wps"

    result = CliRunner().invoke(main, ["plot", "domain-nml", str(namelist), "--hide_lines"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "domain.png").stat().st_size > 0

    result = CliRunner().invoke(main, ["plot", "domain-nml", str(namelist), "-s", "other.png"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "other.png").exists()

    result = CliRunner().invoke(main, ["plot", "domain-nml", "--help"])
    assert result.exit_code == 0
    assert "domain-nml NAMELIST" in result.output and "--info" in result.output  # ours + plot_domains.py options

    result = CliRunner().invoke(main, ["plot", "domain-nml", "-i", "15"])
    assert result.exit_code != 0
    assert "must be the first argument" in result.output
