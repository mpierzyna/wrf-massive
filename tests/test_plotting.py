# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless, no display needed

import matplotlib.dates as mdates
import numpy as np
import xarray as xr
from click.testing import CliRunner

from wrf_massive.cli import main
from wrf_massive.base import Simulation
from wrf_massive.plotting import plot_domain_geo_em, plot_sim_timeline


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
    # output is auto-derived next to the geo_em file as <stem>_domains.png
    out = tmp_path / "geo_em.d01_domains.png"
    assert out.exists() and out.stat().st_size > 0


def test_cli_domain_nml(tmp_path, monkeypatch):
    """`plot domain-nml NAMELIST` runs the vendored WPS plot_domains.py, saving to domains.png by default and passing
    further options through."""
    import wrf_massive
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)  # no Natural Earth downloads
    monkeypatch.chdir(tmp_path)
    namelist = pathlib.Path(wrf_massive.__file__).parent / "templates" / "workspace" / "namelist.tmpl.wps"

    result = CliRunner().invoke(main, ["plot", "domain-nml", str(namelist), "--hide_lines"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "domains.png").stat().st_size > 0

    result = CliRunner().invoke(main, ["plot", "domain-nml", str(namelist), "-s", "other.png"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "other.png").exists()

    result = CliRunner().invoke(main, ["plot", "domain-nml", "--help"])
    assert result.exit_code == 0
    assert "domain-nml NAMELIST" in result.output and "--info" in result.output  # ours + plot_domains.py options

    result = CliRunner().invoke(main, ["plot", "domain-nml", "-i", "15"])
    assert result.exit_code != 0
    assert "must be the first argument" in result.output


def _sim(begin, end, sim_dir):
    return Simulation(begin=begin, end=end, warmup_h=12, sim_dir=sim_dir, settings={})


SIMS_MODULE = """
from wrf_massive.base import Simulation

sim_a = Simulation(begin="2020-01-01", end="2020-01-03", sim_dir="sim_a", settings={})
sims_b = [
    Simulation(begin="2020-01-10", end="2020-01-12", sim_dir="sim_b1", settings={}),
    Simulation(begin="2020-01-20", end="2020-01-25T12:00", sim_dir="sim_b2", settings={}),
]
not_a_sim = 42
"""


def _record_timeline_titles(monkeypatch):
    """Wrap plot_sim_timeline so the titles the CLI draws can be checked."""
    import wrf_massive.plotting

    titles = []

    def wrapper(*args, **kwargs):
        ax = plot_sim_timeline(*args, **kwargs)
        titles.append(ax.get_title())
        return ax

    monkeypatch.setattr(wrf_massive.plotting, "plot_sim_timeline", wrapper)
    return titles


def test_plot_sim_timeline_bounds():
    """Bounds are the earliest begin/latest end (warm-up excluded) unless given; one span per sim plus background."""
    sims = [_sim("2020-01-05", "2020-01-10", "a"), _sim("2020-01-01", "2020-01-03", "b")]
    ax = plot_sim_timeline(sims)
    assert len(ax.patches) == 3
    assert ax.get_title() == ""
    assert plot_sim_timeline(sims, title="my sims").get_title() == "my sims"
    assert ax.get_xlim() == tuple(mdates.date2num([datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 10)]))

    start, end = datetime.datetime(2019, 12, 1), datetime.datetime(2020, 2, 1)
    ax = plot_sim_timeline(sims, start=start, end=end)
    assert ax.get_xlim() == tuple(mdates.date2num([start, end]))


def test_plot_sim_timeline_invalid():
    import pytest

    with pytest.raises(ValueError, match="No simulations"):
        plot_sim_timeline([])
    with pytest.raises(ValueError, match="before its end"):
        plot_sim_timeline([_sim("2020-01-01", "2020-01-03", "a")], start=datetime.datetime(2020, 2, 1))


def test_cli_sim_timeline_from_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    titles = _record_timeline_titles(monkeypatch)
    for s in [_sim("2020-01-01", "2020-01-03", "sim_a"), _sim("2020-01-05", "2020-01-07", "sim_b")]:
        s.to_disk(tmp_path)

    result = CliRunner().invoke(main, ["plot", "sim-timeline", "sim_b/", "sim_a"])
    assert result.exit_code == 0, result.output
    assert "Loaded 2 simulation(s)" in result.output
    assert (tmp_path / "timeline_sim_a...sim_b.png").stat().st_size > 0
    assert titles[-1] == "sim_a...sim_b (n=2 sims, 4 days)"  # sorted dir names

    result = CliRunner().invoke(main, ["plot", "sim-timeline", "sim_a"])
    assert result.exit_code == 0, result.output
    assert titles[-1] == "sim_a (n=1 sims, 2 days)"
    assert (tmp_path / "timeline_sim_a.png").exists()

    (tmp_path / "not_a_sim").mkdir()
    result = CliRunner().invoke(main, ["plot", "sim-timeline", "sim_a", "not_a_sim"])
    assert result.exit_code != 0
    assert "no simulation.yaml" in result.output

    result = CliRunner().invoke(main, ["plot", "sim-timeline"])
    assert result.exit_code != 0
    assert "simulation directory" in result.output


def test_cli_sim_timeline_from_module(tmp_path, monkeypatch):
    """With -m, TARGETS are variable names (a Simulation or list of them) whose sims are combined into one plot."""
    monkeypatch.chdir(tmp_path)
    titles = _record_timeline_titles(monkeypatch)
    (tmp_path / "sims_timeline_test.py").write_text(SIMS_MODULE)
    args = ["plot", "sim-timeline", "-m", "sims_timeline_test.py"]

    result = CliRunner().invoke(main, [*args, "sim_a", "sims_b", "--start", "2019-12-25"])
    assert result.exit_code == 0, result.output
    assert "Found 1 simulation(s) in variable 'sim_a'" in result.output
    assert "Found 2 simulation(s) in variable 'sims_b'" in result.output
    assert (tmp_path / "timeline_sims_timeline_test_sim_a_sims_b.png").stat().st_size > 0
    assert not (tmp_path / "sim_a").exists()  # nothing is initialized on disk
    assert titles[-1] == "sims_timeline_test.py: sim_a, sims_b (n=3 sims, 9.5 days)"

    result = CliRunner().invoke(main, [*args, "missing"])
    assert result.exit_code != 0
    assert "Variable 'missing' not found" in result.output

    result = CliRunner().invoke(main, [*args, "not_a_sim"])
    assert result.exit_code != 0
    assert "must hold a Simulation" in result.output

    result = CliRunner().invoke(main, [*args, "sim_a", "--start", "2021-01-01"])
    assert result.exit_code != 0
    assert "before its end" in result.output
