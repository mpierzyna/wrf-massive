# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""ZarrStore: init-domain, writing (serial, parallel, resumed, ingested), checks, encodings, reading, copies, CLI.

Most tests share one written store (`written`, built once): two sims of the same wrfout fixture, one post-processed
serially, one in parallel, with lossy encodings for some variables, plus the netCDF output of the same stage as
reference. Tests that change a store work on their own one or on a copy.
"""

import datetime
import multiprocessing
import os
import pathlib
import shutil
import sys
import textwrap
import types

import numpy as np
import pydantic
import pytest
import xarray as xr
from click.testing import CliRunner
from fixtures import WRFOUT_DIR, WRFOUT_EXTENDED_DIR

pytest.importorskip("wrf", reason="wrf-python is not installed (`postproc` extra)")
pytest.importorskip("zarr", reason="zarr is not installed (`store` extra)")

from wrf_massive.base import Resources, Simulation
from wrf_massive.cli import main
from wrf_massive.stages.postproc import PostProcStage
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.store.ingest import IngestStage
from wrf_massive.store.netcdf import NetCDFStore
from wrf_massive.store.zarr_store import BitRound, FixedScaleOffset, Lossless, ZarrStore
from wrf_massive.store.zarr_store import layout as L
from wrf_massive.store.zarr_store.init import init_domain
from wrf_massive.store.zarr_store.pull import pull
from wrf_massive.store.zarr_store.read import hp_index, open_domain, select_cells
from wrf_massive.store.zarr_store.verify import verify

VARS = ["w", "u_met", "v_met", "th", "LANDMASK"]  # + z, HGT, p (always) and ct2, cn2 (Cn2 stage)
STATIC = ["HGT", "LANDMASK"]
ENCODING = {"th": FixedScaleOffset(offset=300, step=1e-3), "cn2": BitRound(keepbits=10), "w": BitRound(keepbits=12)}
RESOURCES = Resources(n_tasks=2, cpus_per_task=1, mem_per_cpu="1G")
N_LEAD = 25  # 1 day, hourly, both ends; the fixture's 2 files hold lead times 0, 1, 12, 13


def _sim(root: pathlib.Path, name: str, fixture_dir: pathlib.Path = WRFOUT_DIR, shift_h: int = 0) -> Simulation:
    """Sim `name` (its dir name) with a copy of a fixture's wrfout files, beginning `shift_h` after the first file."""
    sim_dir = root / name
    (sim_dir / "wrfout").mkdir(parents=True)
    for f in fixture_dir.glob("wrfout_aux_d01_*"):
        shutil.copy(f, sim_dir / "wrfout")
    first = sorted((sim_dir / "wrfout").glob("wrfout_aux_d01_*"))[0]
    begin = datetime.datetime.strptime(first.name[15:34], "%Y-%m-%d_%H:%M:%S") + datetime.timedelta(hours=shift_h)
    return Simulation(
        sim_dir=sim_dir, settings={"k": name}, warmup_h=12, begin=begin, end=begin + datetime.timedelta(days=1)
    )


def _store(root: pathlib.Path, domain: str = "d", **kwargs) -> ZarrStore:
    return ZarrStore(path=root / "s.zarr", domain=domain, tile=(2, 2), static_vars=STATIC, **kwargs)


def _stage(store: ZarrStore | NetCDFStore, stage_cls=Cn2PostProcStage, **kwargs):
    defaults = {"work_dir": "pp", "wrfout_dir": "wrfout", "domain": 1, "extract_vars": VARS}
    return stage_cls(store=store, **(defaults | kwargs))


def _open(store: ZarrStore) -> xr.Dataset:
    return open_domain(store.abs_path, store.domain, sort=False, chunks=None)  # no dask: tiny chunks, fast


def _leads(ref: xr.Dataset, s: Simulation) -> np.ndarray:
    return ((ref["time"].values - np.datetime64(s.begin)) // np.timedelta64(1, "h")).astype(int)


def _assert_matches(store: ZarrStore, idx: int, s: Simulation, refs: list[xr.Dataset]):
    """Sim `idx` of the store holds the netCDF reference: lossless variables exactly, encoded ones within bounds."""
    ds = _open(store).drop_vars("time")
    for ref in refs:
        got = ds.isel(sim=idx, lead_time=_leads(ref, s))
        for v in ref.data_vars:
            x = ref[v].values[0] if v in STATIC else ref[v].values
            y = (ds[v] if v in STATIC else got[v]).values
            enc = store.encoding_of(v)
            if isinstance(enc, Lossless):
                np.testing.assert_array_equal(y, x, err_msg=v)
            elif isinstance(enc, BitRound):
                nz = x != 0
                assert (np.abs(y[nz] - x[nz]) / np.abs(x[nz])).max() <= 2.0 ** -(enc.keepbits + 1), v
            else:  # FixedScaleOffset: step / 2, plus float32 resolution at 300 K
                assert np.abs(y - x).max() <= enc.step / 2 + 1e-4, v


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> types.SimpleNamespace:
    """A store with 2 sims (serial / parallel), and the netCDF reference in another workspace's layout
    (`<ref_root>/sim_1/4_postproc`) plus an old single-frame file with a scalar `time` at the end."""
    root = tmp_path_factory.mktemp("written")
    s1, s2 = _sim(root, "sim_1"), _sim(root, "sim_2")
    store = _store(root, encoding=ENCODING)
    stage = _stage(store)
    init_domain(stage, [s1, s2])
    stage.run(s1)
    stage.model_copy(update={"run_parallel": True, "resources": RESOURCES}).run(s2)

    ref_root = root / "old_ws"
    ref_stage = _stage(NetCDFStore(file_suffix="cn2", compression=False, root=ref_root), work_dir="4_postproc")
    ref_stage.run(s1)
    files = ref_stage.store.outputs(ref_stage, s1)
    refs = [xr.open_dataset(f).load() for f in files]
    last = refs[-1].isel(time=0).assign_coords(time=np.datetime64(s1.end, "ns"))  # scalar time, as before 2.3.0
    last.encoding = {}
    last.to_netcdf(files[0].parent / f"wrfout_aux_d01_{s1.end:%Y-%m-%d_%H-%M-%S}_cn2.nc", engine="h5netcdf")
    return types.SimpleNamespace(root=root, store=store, stage=stage, s1=s1, s2=s2, refs=refs, ref_root=ref_root)


def _copy(written, tmp_path) -> pathlib.Path:
    """A copy of the written store, for tests that change it."""
    return pathlib.Path(shutil.copytree(written.store.abs_path, tmp_path / "s.zarr"))


# --- init-domain ---------------------------------------------------------------------------------------------------


def test_init_domain_layout(tmp_path):
    """Creating a domain writes only metadata and coordinates, laid out as in DESIGN.md; appending is idempotent."""
    stage, s1, s2 = _stage(_store(tmp_path)), _sim(tmp_path, "sim_1"), _sim(tmp_path, "sim_2")
    assert "creating" in init_domain(stage, [s1])[0]
    assert init_domain(stage, [s1, s2])[-1] == "Appending 1 sim(s) (1 already present: sim_1)"

    ds = _open(stage.store)
    assert dict(ds.sizes) == {"sim": 2, "lead_time": N_LEAD, "bottom_top": 100, "y": 5, "x": 5}
    assert ds["cn2"].dims == L.DIMS_3D and ds["cn2"].dtype == np.float32
    assert ds["lead_time"].dtype == "timedelta64[s]" and ds["lead_time"].values[1] == np.timedelta64(1, "h")
    assert list(ds["sim_name"].values) == ["sim_1", "sim_2"] and ds["time"].values[1, 0] == np.datetime64(s2.begin)
    assert set(ds.coords) == {"lead_time", "time", "XLAT", "XLONG"}
    assert ds.attrs["shard_len"] == 2 and ds.attrs["static_vars"] == STATIC and not ds["written"].values.any()
    # No data chunks yet
    assert sorted(p.name for p in (stage.store.abs_path / "d" / "cn2").iterdir()) == ["zarr.json"]


def test_init_domain_reports_and_errors(tmp_path):
    """Dry run, refused layouts, encodings shared across domains (reported, warned) and sims of another length."""
    stage, s = _stage(_store(tmp_path)), _sim(tmp_path, "sim_1")
    assert any("does not exist -> creating" in line for line in init_domain(stage, [s], dry_run=True))
    assert not stage.store.abs_path.exists()  # nothing written, not even the lock

    empty = Simulation(sim_dir=tmp_path / "empty", settings={}, warmup_h=0, begin=s.begin, end=s.end)
    with pytest.raises(ValueError, match="none of the 1 given sim"):
        init_domain(stage, [empty])
    with pytest.raises(ValueError, match="Destagger"):  # EL_PBL read raw, on bottom_top_stag
        raw = ZarrStore(path=tmp_path / "x.zarr", domain="d", static_vars=["HGT"])
        init_domain(_stage(raw, stage_cls=PostProcStage, extract_vars=["EL_PBL"]), [s])
    with pytest.raises(ValueError, match="static_vars"):
        init_domain(_stage(ZarrStore(path=tmp_path / "x.zarr", domain="d", static_vars=["NOPE"])), [s])
    with pytest.raises(ValueError, match="encoding given for static_vars"):
        init_domain(_stage(_store(tmp_path, encoding={"HGT": Lossless()})), [s])

    init_domain(stage, [s])
    # A project-wide encoding: entries for absent variables are reported, differences to other domains warned about
    other = _stage(_store(tmp_path, domain="d2", encoding={"QSQ": BitRound(keepbits=10), "w": BitRound(keepbits=8)}))
    report = init_domain(other, [s])
    assert any("not in this domain's output: ['QSQ']" in line for line in report)
    assert any(line.startswith("WARNING: 'w'") and "domain 'd'" in line for line in report)

    longer = Simulation(
        sim_dir=tmp_path / "long", settings={}, warmup_h=12, begin=s.begin, end=s.end + (s.end - s.begin)
    )
    with pytest.raises(ValueError, match="lead times"):
        init_domain(stage, [longer])


# --- Writing -------------------------------------------------------------------------------------------------------


def test_postproc_into_store(written):
    """Serial and parallel runs write what the netCDF store writes (encoded variables within their bounds) into the
    right [sim, lead_time], record their encodings and commit exactly the written frames."""
    st, s1, s2 = written.stage, written.s1, written.s2
    assert st.is_done(s1) and st.is_done(s2)
    _assert_matches(written.store, 0, s1, written.refs)
    ds = _open(written.store)
    data = [v for v in ds.data_vars if v not in ("sim_name", "sim_settings", "sim_begin")]
    xr.testing.assert_identical(ds[data].isel(sim=0).drop_vars("time"), ds[data].isel(sim=1).drop_vars("time"))
    assert ds["th"].attrs["encoding"] == {"kind": "fixed_scale_offset", "offset": 300.0, "step": 0.001}
    assert ds["p"].attrs["encoding"] == {"kind": "lossless"}

    expected = np.zeros(N_LEAD, bool)
    expected[np.concatenate([_leads(r, s1) for r in written.refs])] = True
    assert (ds["written"].values == expected).all()


def test_resume_per_frame(tmp_path, monkeypatch, written):
    """A crash leaves the frames written so far committed; a re-run processes only the missing frames."""
    stage, s = _stage(_store(tmp_path, encoding=ENCODING)), _sim(tmp_path, "sim_1")
    init_domain(stage, [s])
    calls, original = [], PostProcStage._process_frame

    def process(self, f, i, crash_at=None):
        calls.append((f.name, i))
        if len(calls) == crash_at:
            raise RuntimeError("killed")
        return original(self, f, i)

    monkeypatch.setattr(PostProcStage, "_process_frame", lambda self, f, i: process(self, f, i, crash_at=3))
    with pytest.raises(RuntimeError, match="killed"):
        stage.run(s)
    assert _open(stage.store)["written"].values.sum() == 2 and not stage.is_done(s)

    calls.clear()
    monkeypatch.setattr(PostProcStage, "_process_frame", process)
    stage.run(s)
    assert len(calls) == 2 and stage.is_done(s)  # only the second file's two frames
    _assert_matches(stage.store, 0, s, written.refs)
    calls.clear()
    stage.run(s)  # complete: nothing to do
    assert not calls


def test_ingest(tmp_path, written):
    """Ingesting the netCDF output of another workspace gives the same arrays as post-processing into the store,
    also for an old single-frame file with a scalar `time`; a re-run writes nothing."""
    s = written.s1
    store = written.store.model_copy(update={"path": tmp_path / "ingested.zarr"})
    ingest = IngestStage(work_dir="ingest", src=NetCDFStore(file_suffix="cn2", root=written.ref_root), store=store)
    assert any("creating from template" in line and "4_postproc" in line for line in init_domain(ingest, [s]))
    assert not ingest.is_done(s)
    ingest.run(s)
    assert ingest.is_done(s)

    a, b = _open(written.store).isel(sim=[0]), _open(store)
    for v in a.data_vars:
        if v != "written":
            sel = {"lead_time": slice(0, N_LEAD - 1)} if "lead_time" in a[v].dims else {}
            xr.testing.assert_equal(a[v].isel(sel), b[v].isel(sel))
    last = NetCDFStore.read_frames(sorted((written.ref_root / "sim_1" / "4_postproc").glob("*.nc"))[-1])
    assert b["written"].values[0, -1]  # the scalar-time frame
    np.testing.assert_allclose(b["cn2"].values[0, -1], last["cn2"].values[0], rtol=2.0**-11)

    mtimes = {f: f.stat().st_mtime_ns for f in store.abs_path.rglob("c.*")}
    ingest.run(s)
    assert {f: f.stat().st_mtime_ns for f in store.abs_path.rglob("c.*")} == mtimes


def test_session_checks(tmp_path):
    """Nothing is processed unless the store, the sim and the inputs fit, and no other job writes the sim."""
    stage, s = _stage(_store(tmp_path)), _sim(tmp_path, "sim_1")
    with pytest.raises(RuntimeError, match="init-domain"):
        stage.run(s)  # no domain yet
    init_domain(stage, [s])

    with pytest.raises(RuntimeError, match="not initialized"):
        stage.run(_sim(tmp_path, "sim_2"))
    with pytest.raises(RuntimeError, match=r"only in the stage \['rh'\]"):
        _stage(stage.store, extract_vars=[*VARS, "rh"]).run(s)
    with pytest.raises(RuntimeError, match="Encodings differ"):
        _stage(stage.store.model_copy(update={"encoding": {"cn2": BitRound(keepbits=8)}})).run(s)

    other = _sim(tmp_path, "sim_other", WRFOUT_EXTENDED_DIR)  # another grid, appended to this domain
    init_domain(stage, [other])
    with pytest.raises(RuntimeError, match="differs from the store's"):
        stage.run(other)

    late = _sim(tmp_path, "sim_late", shift_h=1)  # its first file then has a frame before begin (spin-up)
    init_domain(stage, [late])
    with pytest.raises(RuntimeError, match="before Simulation.begin"):
        stage.model_copy(update={"discard_warmup": False}).run(late)

    # One job per sim: the lock is per process, so take it from another process, as a second job would
    with L.exclusive_lock(stage.get_work_dir(s) / ".zarr_store.lock", "test"):
        with pytest.raises(RuntimeError, match="locked"), multiprocessing.get_context("spawn").Pool(1) as pool:
            pool.apply(_run_stage, (stage, s))
    assert not _open(stage.store)["written"].values.any()  # none of the failed runs wrote anything


def _run_stage(stage, s):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    stage.run(s)


# --- Reading, copies, verify ---------------------------------------------------------------------------------------


def test_open_domain_sorting_and_healpix(tmp_path):
    """Sims are sorted by begin, whatever their append order; HEALPix cells at any order from `hp_ipix`."""
    stage, late, early = _stage(_store(tmp_path)), _sim(tmp_path, "sim_late"), _sim(tmp_path, "sim_early")
    shift = datetime.timedelta(days=3)
    init_domain(stage, [late])
    init_domain(stage, [early.model_copy(update={"begin": early.begin - shift, "end": early.end - shift})])

    path = stage.store.abs_path
    assert list(open_domain(path, "d")["sim_name"].values) == ["sim_early", "sim_late"]
    assert list(open_domain(path, "d", sort=False)["sim_name"].values) == ["sim_late", "sim_early"]
    with pytest.raises(KeyError, match="not in"):
        open_domain(path, "nope")

    import healpy

    ds = open_domain(path, "d")
    ipix10 = healpy.ang2pix(
        2**10, ds["XLONG"].values.astype("f8"), ds["XLAT"].values.astype("f8"), nest=True, lonlat=True
    )
    np.testing.assert_array_equal(hp_index(ds, 10).values, ipix10)
    cell = ipix10.ravel()[0]
    assert (select_cells(ds, [cell], 10).values == (ipix10 == cell)).all()
    with pytest.raises(ValueError, match="order"):
        hp_index(ds, 24)


def test_pull_partial_copy(tmp_path, written):
    """A pull copies metadata and the selected variables/sims only, never temp files; the reader exposes only what
    the copy holds; pulls add up."""
    src = _copy(written, tmp_path)
    (src / "d" / "cn2" / "c.0.0.0.0.0.deadbeef.partial").write_bytes(b"x")  # interrupted write
    dst = tmp_path / "copy.zarr"

    assert pull(str(src), dst, ["d"], ["cn2", "w"], ["sim_1"], dry_run=True)[0].startswith("Would copy")
    assert not (dst / "d" / "cn2").exists()
    assert pull(str(src), dst, ["d"], ["cn2", "w"], ["sim_1"])[-1] == "d: 2 of 8 variable(s), 1 of 2 sim(s)"
    assert not list(dst.rglob("*.partial")) and not (dst / "d" / "p").exists()
    assert sorted(f.name for f in (dst / "d" / "cn2").iterdir()) == ["c.0.0.0.0.0", "c.0.6.0.0.0", "zarr.json"]

    ds = open_domain(dst, "d")
    assert list(ds["sim_name"].values) == ["sim_1"] and {"cn2", "w"} <= set(ds.data_vars) and "p" not in ds
    xr.testing.assert_equal(ds["cn2"].load(), open_domain(src, "d")["cn2"].isel(sim=[0]).load())

    pull(str(src), dst, ["d"], ["cn2"], ["sim_2"])
    assert list(open_domain(dst, "d")["sim_name"].values) == ["sim_1", "sim_2"]
    with pytest.raises(KeyError, match="No sim"):
        pull(str(src), dst, ["d"], sims=["sim_9*"])
    with pytest.raises(KeyError, match="no variable"):
        pull(str(src), dst, ["d"], variables=["nope"])


def test_verify(tmp_path, written):
    path = _copy(written, tmp_path)
    L.open_domain(path, "d", mode="r+")[L.WRITTEN][1, 0] = False
    (path / "d" / "w" / "c.1.0.0.0.0.abc.partial").write_bytes(b"x")

    report = verify(path)
    line = next(r for r in report if r.strip().startswith("d "))
    assert "2 sims  7/50 frames written" in line and "incomplete: 2 sims (sim_1, sim_2)" in line  # 4 of 25 each
    assert "stray .partial files: 1" in report[-2]
    assert any("1 sims  3/25 frames written" in r for r in verify(path, sims=["sim_2"]))
    assert "removed 1 .partial file(s)" in verify(path, clean=True)[-1] and not list(path.rglob("*.partial"))


# --- Config and CLI ------------------------------------------------------------------------------------------------


def test_config_validation():
    """Store configs and encodings validate and serialize like any config; FixedScaleOffset refuses what int32
    cannot hold; IngestStage shares the parallel options and checks of PostProcStage (FileParallelStage)."""
    store = ZarrStore.model_validate(
        {
            "path": "s.zarr",
            "domain": "d",
            "encoding": {"th": {"kind": "fixed_scale_offset", "offset": 300, "step": 1e-3}},
        }
    )
    assert store.encoding_of("th") == FixedScaleOffset(offset=300, step=1e-3) and store.encoding_of("p") == Lossless()
    assert ZarrStore.model_validate(store.model_dump()) == store
    for invalid in (lambda: BitRound(keepbits=0), lambda: ZarrStore(path="s.zarr", domain="a/b")):
        with pytest.raises(pydantic.ValidationError):
            invalid()

    fso = FixedScaleOffset(offset=300, step=1e-3)
    fso.check("th", np.array([200.0, 800.0], dtype="f4"))
    with pytest.raises(ValueError, match="do not fit int32"):
        fso.check("th", np.array([3e6], dtype="f4"))
    with pytest.raises(ValueError, match="NaN"):
        fso.check("th", np.array([np.nan], dtype="f4"))

    kw = dict(work_dir="i", src=NetCDFStore(), store=store)
    with pytest.raises(pydantic.ValidationError, match="exceeds the allocated CPUs"):
        IngestStage(**kw, run_parallel=True, resources=RESOURCES, n_workers=3)
    with pytest.raises(pydantic.ValidationError, match="requires 'resources'"):
        IngestStage(**kw, run_parallel=True)


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


def test_cli(tmp_path, monkeypatch, written):
    """`store init-domain` adds the domain of a workspace stage (relative store path: to the workspace root), with
    --debug only reporting; `run` fills it with spawned workers, which import the workspace's pipeline.py
    themselves; `store verify --path` works without a workspace."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    cwd = os.getcwd()
    try:
        (tmp_path / "env.yaml").write_text("pipeline: p\n")
        (tmp_path / "pipeline.py").write_text(textwrap.dedent(PIPELINE_PY))
        s = _sim(tmp_path, "sim_1")
        s.to_disk(tmp_path)
        os.chdir(s.sim_dir)  # any dir inside the workspace
        invoke = lambda *args: CliRunner().invoke(main, [str(a) for a in args])  # noqa: E731

        result = invoke("--debug", "store", "init-domain", "--stage", "cn2", ".")
        assert result.exit_code == 0 and "does not exist -> creating" in result.output, result.output
        assert not (tmp_path / "store.zarr").exists()
        # The CLI changed into the workspace root, so `.` would now be the root: give the sim dir explicitly
        assert invoke("store", "init-domain", "--stage", "cn2", s.sim_dir).exit_code == 0
        assert invoke("run", "--stages", "cn2", s.sim_dir).exit_code == 0
        assert xr.open_zarr(tmp_path / "store.zarr" / "ws", consolidated=True)["written"].values.sum() == 4
        result = invoke("store", "init-domain", "--stage", "nope", s.sim_dir)
        assert result.exit_code != 0 and "Unknown stage 'nope'" in result.output

        result = invoke("store", "verify", "--path", written.store.abs_path)
        assert result.exit_code == 0 and "8/50 frames written" in result.output
    finally:
        os.chdir(cwd)
        sys.modules.pop("pipeline", None)
