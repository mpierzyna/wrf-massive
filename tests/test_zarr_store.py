# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""ZarrStore: init-domain, postproc writing (serial, parallel, resumed), checks, encodings and the CLI."""

import datetime
import os
import pathlib
import shutil
import sys
import textwrap

import numpy as np
import pydantic
import pytest
import xarray as xr
from click.testing import CliRunner
from fixtures import WRFOUT_DIR, WRFOUT_DIRS, WRFOUT_EXTENDED_DIR

pytest.importorskip("wrf", reason="wrf-python is not installed (`postproc` extra)")
pytest.importorskip("zarr", reason="zarr is not installed (`store` extra)")

from wrf_massive.base import Resources, Simulation
from wrf_massive.cli import main
from wrf_massive.stages.postproc import PostProcStage
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.store.netcdf import NetCDFStore
from wrf_massive.store.zarr_store import BitRound, FixedScaleOffset, Lossless, ZarrStore
from wrf_massive.store.zarr_store import layout as L
from wrf_massive.store.zarr_store.init import init_domain

VARS = ["w", "u_met", "v_met", "th", "LANDMASK"]
STATIC = ["HGT", "LANDMASK"]
RESOURCES = Resources(n_tasks=2, cpus_per_task=1, mem_per_cpu="1G")


def _sim(tmp_path: pathlib.Path, name: str, fixture_dir: pathlib.Path = WRFOUT_DIR, shift_h: int = 0) -> Simulation:
    """Sim `name` (its dir name) with a copy of a fixture's wrfout files, beginning `shift_h` after the first file."""
    sim_dir = tmp_path / name
    (sim_dir / "wrfout").mkdir(parents=True)
    for f in fixture_dir.glob("wrfout_aux_d01_*"):
        shutil.copy(f, sim_dir / "wrfout")
    first = sorted((sim_dir / "wrfout").glob("wrfout_aux_d01_*"))[0]
    begin = datetime.datetime.strptime(first.name[15:34], "%Y-%m-%d_%H:%M:%S") + datetime.timedelta(hours=shift_h)
    return Simulation(
        sim_dir=sim_dir, settings={"k": name}, warmup_h=12, begin=begin, end=begin + datetime.timedelta(days=1)
    )


def _stage(tmp_path: pathlib.Path, store=None, stage_cls=Cn2PostProcStage, **kwargs):
    store = store or ZarrStore(path=tmp_path / "s.zarr", domain="d", tile=(2, 2), static_vars=STATIC)
    kwargs = {"extract_vars": VARS, **kwargs}
    return stage_cls(work_dir="pp", wrfout_dir="wrfout", domain=1, store=store, **kwargs)


def _open(stage) -> xr.Dataset:
    return xr.open_zarr(stage.store.abs_path / stage.store.domain, consolidated=True)


def _netcdf_reference(tmp_path: pathlib.Path, stage, s: Simulation) -> list[xr.Dataset]:
    """The same stage writing netCDF: what the store must hold."""
    ref = stage.model_copy(update={"store": NetCDFStore(compression=False), "work_dir": tmp_path / "ref"})
    ref.work_dir.mkdir(exist_ok=True)
    ref.run(s)
    return [xr.open_dataset(f).load() for f in ref.store.outputs(ref, s)]


def _assert_matches(stage, s: Simulation, refs: list[xr.Dataset], idx: int = 0, **tol):
    ds = _open(stage).drop_vars("time")
    for ref in refs:
        lead = (ref["time"].values - np.datetime64(s.begin)) // np.timedelta64(1, "h")
        got = ds.isel(sim=idx, lead_time=lead.astype(int)).rename(lead_time="time", y="south_north", x="west_east")
        for v in ref.data_vars:
            if v in STATIC:
                np.testing.assert_array_equal(ds[v].values, ref[v].values[0], err_msg=v)
            elif tol:
                np.testing.assert_allclose(got[v].values, ref[v].values, err_msg=v, **tol)
            else:
                np.testing.assert_array_equal(got[v].values, ref[v].values, err_msg=v)


# --- init-domain ---------------------------------------------------------------------------------------------------


def test_init_domain_layout(tmp_path):
    """Creating a domain writes only metadata and coordinates, laid out as in DESIGN.md; appending is idempotent."""
    stage, s1, s2 = _stage(tmp_path), _sim(tmp_path, "sim_1"), _sim(tmp_path, "sim_2")
    report = init_domain(stage, [s1])
    assert "creating" in report[0] and "Appending 1 sim(s)" in report[-1]
    report = init_domain(stage, [s1, s2])
    assert report[-1] == "Appending 1 sim(s) (1 already present: sim_1)"

    ds = _open(stage)
    n_lead = 25  # 1 day, hourly, both ends
    assert dict(ds.sizes) == {"sim": 2, "lead_time": n_lead, "bottom_top": 100, "y": 5, "x": 5}
    assert ds["cn2"].dims == L.DIMS_3D and ds["cn2"].dtype == np.float32
    assert ds["lead_time"].dtype == "timedelta64[s]" and ds["lead_time"].values[1] == np.timedelta64(1, "h")
    assert list(ds["sim_name"].values) == ["sim_1", "sim_2"]
    assert ds["time"].values[1, 0] == np.datetime64(s2.begin)
    assert set(ds.coords) == {"lead_time", "time", "XLAT", "XLONG"}
    assert not ds["written"].values.any()
    assert ds.attrs["shard_len"] == 2 and ds.attrs["static_vars"] == STATIC

    # HEALPix index at order 23 of the grid points; coarser orders by bit shift
    import healpy

    lat, lon = ds["XLAT"].values.astype("f8"), ds["XLONG"].values.astype("f8")
    ipix8 = healpy.ang2pix(2**8, lon, lat, nest=True, lonlat=True)
    np.testing.assert_array_equal(ds["hp_ipix"].values >> (2 * (23 - 8)), ipix8)

    # No data chunks yet; shard files are flat (`.` separator)
    domain_dir = stage.store.abs_path / "d"
    assert sorted(p.name for p in (domain_dir / "cn2").iterdir()) == ["zarr.json"]
    assert not (domain_dir / "written" / "c").exists()


def test_init_domain_dry_run_and_errors(tmp_path):
    stage, s = _stage(tmp_path), _sim(tmp_path, "sim_1")
    report = init_domain(stage, [s], dry_run=True)
    assert any("does not exist -> creating" in line for line in report)
    assert not stage.store.abs_path.exists()  # nothing written, not even the lock

    with pytest.raises(ValueError, match="none of the 1 given sim"):
        init_domain(stage, [Simulation(sim_dir=tmp_path / "empty", settings={}, warmup_h=0, begin=s.begin, end=s.end)])
    with pytest.raises(ValueError, match="Destagger"):  # EL_PBL read raw, on bottom_top_stag
        raw = ZarrStore(path=tmp_path / "x.zarr", domain="d", static_vars=["HGT"])
        init_domain(_stage(tmp_path, raw, stage_cls=PostProcStage, extract_vars=["EL_PBL"]), [s])
    with pytest.raises(ValueError, match="static_vars"):
        init_domain(_stage(tmp_path, ZarrStore(path=tmp_path / "x.zarr", domain="d", static_vars=["NOPE"])), [s])
    with pytest.raises(ValueError, match="encoding given for"):
        init_domain(
            _stage(tmp_path, ZarrStore(path=tmp_path / "x.zarr", domain="d", encoding={"NOPE": Lossless()})), [s]
        )

    init_domain(stage, [s])
    longer = Simulation(
        sim_dir=tmp_path / "sim_long", settings={}, warmup_h=12, begin=s.begin, end=s.end + (s.end - s.begin)
    )
    with pytest.raises(ValueError, match="lead times"):
        init_domain(stage, [longer])


# --- Writing -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_dir", WRFOUT_DIRS, ids=lambda d: d.name)
def test_postproc_into_store_matches_netcdf(tmp_path, fixture_dir):
    """Serial and parallel runs write exactly what the netCDF store writes, into the right [sim, lead_time]."""
    stage, s1, s2 = _stage(tmp_path), _sim(tmp_path, "sim_1", fixture_dir), _sim(tmp_path, "sim_2", fixture_dir)
    init_domain(stage, [s1, s2])
    assert not stage.is_done(s1)
    stage.run(s1)
    stage.model_copy(update={"run_parallel": True, "resources": RESOURCES}).run(s2)
    assert stage.is_done(s1) and stage.is_done(s2)

    refs = _netcdf_reference(tmp_path, stage, s1)
    _assert_matches(stage, s1, refs, idx=0)
    _assert_matches(stage, s2, refs, idx=1)  # same files, so same data

    written = _open(stage)["written"].values
    expected = np.zeros(25, bool)
    for ref in refs:
        expected[(ref["time"].values - np.datetime64(s1.begin)) // np.timedelta64(1, "h")] = True
    assert (written == expected).all()


def test_resume_per_frame(tmp_path, monkeypatch):
    """A crash leaves the frames written so far committed; a re-run processes only the missing frames."""
    stage, s = _stage(tmp_path), _sim(tmp_path, "sim_1")
    init_domain(stage, [s])
    calls = []
    original = PostProcStage._process_frame

    def crash_on_third(self, f, i):
        calls.append((f.name, i))
        if len(calls) == 3:
            raise RuntimeError("killed")
        return original(self, f, i)

    monkeypatch.setattr(PostProcStage, "_process_frame", crash_on_third)
    with pytest.raises(RuntimeError, match="killed"):
        stage.run(s)
    assert _open(stage)["written"].values.sum() == 2 and not stage.is_done(s)

    calls.clear()
    monkeypatch.setattr(PostProcStage, "_process_frame", original)
    monkeypatch.setattr(
        PostProcStage, "_process_frame", lambda self, f, i: calls.append((f.name, i)) or original(self, f, i)
    )
    stage.run(s)
    assert len(calls) == 2 and stage.is_done(s)  # only the second file's two frames
    _assert_matches(stage, s, _netcdf_reference(tmp_path, stage, s))

    calls.clear()
    stage.run(s)  # complete: nothing to do
    assert not calls


def test_lossy_encodings(tmp_path):
    """Encodings are applied with their error bounds and recorded on the arrays."""
    enc = {"th": FixedScaleOffset(offset=300, step=1e-3), "cn2": BitRound(keepbits=10), "w": BitRound(keepbits=12)}
    store = ZarrStore(path=tmp_path / "s.zarr", domain="d", tile=(2, 2), static_vars=STATIC, encoding=enc)
    stage, s = _stage(tmp_path, store), _sim(tmp_path, "sim_1")
    init_domain(stage, [s])
    stage.run(s)
    ds = _open(stage)
    assert ds["th"].attrs["encoding"] == {"kind": "fixed_scale_offset", "offset": 300.0, "step": 0.001}
    assert ds["p"].attrs["encoding"] == {"kind": "lossless"}

    for ref in _netcdf_reference(tmp_path, stage, s):
        lead = ((ref["time"].values - np.datetime64(s.begin)) // np.timedelta64(1, "h")).astype(int)
        got = ds.isel(sim=0, lead_time=lead)
        assert np.abs(got["th"].values - ref["th"].values).max() <= 0.5e-3 + 1e-4  # + float32 resolution at 300 K
        for v, n in [("cn2", 10), ("w", 12)]:
            x, y = ref[v].values, got[v].values
            nz = x != 0
            assert (np.abs(y[nz] - x[nz]) / np.abs(x[nz])).max() <= 2.0 ** -(n + 1)
        np.testing.assert_array_equal(got["p"].values, ref["p"].values)  # lossless


def test_encoding_config():
    """Encodings validate and serialize like any config; FixedScaleOffset refuses what int32 cannot hold."""
    store = ZarrStore.model_validate(
        {
            "path": "s.zarr",
            "domain": "d",
            "encoding": {"th": {"kind": "fixed_scale_offset", "offset": 300, "step": 1e-3}},
        }
    )
    assert store.encoding_of("th") == FixedScaleOffset(offset=300, step=1e-3) and store.encoding_of("p") == Lossless()
    assert ZarrStore.model_validate(store.model_dump()) == store
    with pytest.raises(pydantic.ValidationError):
        BitRound(keepbits=0)
    with pytest.raises(pydantic.ValidationError):
        ZarrStore(path="s.zarr", domain="a/b")
    fso = FixedScaleOffset(offset=300, step=1e-3)
    fso.check("th", np.array([200.0, 800.0], dtype="f4"))
    with pytest.raises(ValueError, match="do not fit int32"):
        fso.check("th", np.array([3e6], dtype="f4"))
    with pytest.raises(ValueError, match="NaN"):
        fso.check("th", np.array([np.nan], dtype="f4"))


# --- Checks before writing -----------------------------------------------------------------------------------------


def test_session_checks(tmp_path):
    """Nothing is processed unless the store, the sim and the inputs fit, and no other job writes the sim."""
    stage, s = _stage(tmp_path), _sim(tmp_path, "sim_1")
    with pytest.raises(RuntimeError, match="init-domain"):
        stage.run(s)  # no domain yet
    init_domain(stage, [s])

    with pytest.raises(RuntimeError, match="not initialized"):
        stage.run(_sim(tmp_path, "sim_2"))
    with pytest.raises(RuntimeError, match=r"only in the stage \['rh'\]"):
        _stage(tmp_path, extract_vars=[*VARS, "rh"]).run(s)
    enc = stage.store.model_copy(update={"encoding": {"cn2": BitRound(keepbits=8)}})
    with pytest.raises(RuntimeError, match="Encodings differ"):
        _stage(tmp_path, enc).run(s)

    # Another grid: sim of the other fixture appended to this domain
    other = _sim(tmp_path, "sim_other", WRFOUT_EXTENDED_DIR)
    init_domain(stage, [other])
    with pytest.raises(RuntimeError, match="differs from the store's"):
        stage.run(other)

    # Frames before Simulation.begin (spin-up) are refused
    late = _sim(tmp_path, "sim_late", shift_h=1)
    init_domain(stage, [late])
    with pytest.raises(RuntimeError, match="before Simulation.begin"):
        stage.model_copy(update={"discard_warmup": False}).run(late)

    # One job per sim
    with L.exclusive_lock(stage.get_work_dir(s) / ".zarr_store.lock", "test"):
        os.environ.pop("_", None)
        with pytest.raises(RuntimeError, match="locked"):
            # the lock is per process: take it from another process, as a second job would
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(1) as pool:
                pool.apply(_run_stage, (stage, s))
    assert not _open(stage)["written"].values.any()  # none of the failed runs wrote anything


def _run_stage(stage, s):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    stage.run(s)


def test_cross_domain_encoding_warning(tmp_path):
    s = _sim(tmp_path, "sim_1")
    init_domain(_stage(tmp_path), [s])
    store = ZarrStore(
        path=tmp_path / "s.zarr", domain="d2", tile=(2, 2), static_vars=STATIC, encoding={"w": BitRound(keepbits=8)}
    )
    report = init_domain(_stage(tmp_path, store), [s])
    assert any(line.startswith("WARNING: 'w'") and "domain 'd'" in line for line in report)


# --- CLI -----------------------------------------------------------------------------------------------------------

PIPELINE_PY = """
from wrf_massive.base import Pipeline, Resources
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.store.zarr_store import ZarrStore

p = Pipeline(
    cn2=Cn2PostProcStage(
        work_dir="pp", wrfout_dir="wrfout", domain=1, extract_vars=["w"],
        store=ZarrStore(path="store.zarr", domain="ws", tile=(2, 2), static_vars=["HGT"]),
        run_parallel=True, resources=Resources(n_tasks=1, cpus_per_task=2, mem_per_cpu="1G"),
    ),
)
"""


def test_cli_init_domain(tmp_path, monkeypatch):
    """`store init-domain` adds the domain of a workspace stage; with --debug it only reports. The store path is
    relative to the workspace root. `run` then fills it with spawned workers, which import the workspace's
    pipeline.py themselves."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    cwd = os.getcwd()
    try:
        (tmp_path / "env.yaml").write_text("pipeline: p\n")
        (tmp_path / "pipeline.py").write_text(textwrap.dedent(PIPELINE_PY))
        s = _sim(tmp_path, "sim_1")
        s.to_disk(tmp_path)
        os.chdir(s.sim_dir)  # any dir inside the workspace

        result = CliRunner().invoke(main, ["--debug", "store", "init-domain", "--stage", "cn2", "."])
        assert result.exit_code == 0, result.output
        assert "does not exist -> creating" in result.output and not (tmp_path / "store.zarr").exists()

        # The CLI changed into the workspace root, so `.` would now be the root: give the sim dir explicitly
        result = CliRunner().invoke(main, ["store", "init-domain", "--stage", "cn2", str(s.sim_dir)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "store.zarr" / "ws" / "zarr.json").exists()

        result = CliRunner().invoke(main, ["run", "--stages", "cn2", str(s.sim_dir)])
        assert result.exit_code == 0, result.output
        written = xr.open_zarr(tmp_path / "store.zarr" / "ws", consolidated=True)["written"].values
        assert written.sum() == 4  # 2 files x 2 frames

        result = CliRunner().invoke(main, ["store", "init-domain", "--stage", "nope", str(s.sim_dir)])
        assert result.exit_code != 0 and "Unknown stage 'nope'" in result.output
    finally:
        os.chdir(cwd)
        sys.modules.pop("pipeline", None)
