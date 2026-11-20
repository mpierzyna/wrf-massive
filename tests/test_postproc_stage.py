# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import pathlib
import re
import warnings

import netCDF4
import numpy as np
import pydantic
import pytest
import xarray as xr
from fixtures import WRFOUT_DIR, WRFOUT_DIRS

pytest.importorskip("wrf", reason="wrf-python is not installed (`postproc` extra)")


from wrf_massive.base import Resources, Simulation
from wrf_massive.stages.postproc import PostProcFn, PostProcStage
from wrf_massive.stages.postproc.base import TVarList
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage, fn_ct2_cn2
from wrf_massive.stages.postproc.iofields import parse_myoutfields
from wrf_massive.stages.postproc.precip import fn_int_water_paths
from wrf_massive.stages.postproc.variables import DERIVED_VARS, REQUIRED_WRF_FIELDS, var_name, wrf_fields_of
from wrf_massive.stages.postproc.wrfout import load_wrfout
from wrf_massive.store.netcdf import NetCDFStore

# --- Helpers over the wrfout fixtures -------------------------------------------------------------------------------
# The fixtures differ in which fields WRF wrote (see fixtures.py). Tests take what they need from the fixtures'
# contents instead of naming a particular setup, so they hold for any WRF output configuration.

_STAGE_KWARGS = dict(work_dir="/tmp", wrfout_dir="x", domain=1)
_RESOURCES = Resources(n_tasks=2, cpus_per_task=1, mem_per_cpu="1G")
_FN_A = PostProcFn(fn=lambda ds: {"a": ds["tk"] * 2}, requires=["tk"], returns=["a"])
_FN_B = PostProcFn(fn=lambda ds: {"b": ds["a"] + 1}, requires=["a"], returns=["b"])


def _first_file(fixture_dir: pathlib.Path = WRFOUT_DIR) -> pathlib.Path:
    return sorted(fixture_dir.glob("wrfout_aux_d01_*"))[0]


def _fixture_fields(fixture_dir: pathlib.Path = WRFOUT_DIR) -> list:
    """WRF fields in a wrfout fixture (what the WRF run behind it wrote to its aux stream)."""
    with netCDF4.Dataset(_first_file(fixture_dir)) as f:
        return [v for v in f.variables if v != "Times"]


def _readable_vars(fixture_dir: pathlib.Path) -> TVarList:
    """Everything a fixture provides: the WRF fields and derived variables it has all needed fields for.

    EL_PBL is requested destaggered, as `fn_ct2_cn2` needs it. The horizontally staggered U/V are left out: wrf-python
    reads them raw only with their own coordinates (XLAT_U/XLONG_U, XLAT_V/XLONG_V), which wrfout_aux files lack.
    Their earth-relative versions `u_met`/`v_met` are included as derived variables.
    """
    fields = set(_fixture_fields(fixture_dir))
    names = [*sorted(fields - {"U", "V"}), *DERIVED_VARS]
    readable = [n for n in names if set(wrf_fields_of(n)) <= fields]
    return [("EL_PBL", "bottom_top_stag") if n == "EL_PBL" else n for n in readable]


def _write_myoutfields(tmp_path, fields, stream: int = 7) -> pathlib.Path:
    path = tmp_path / "myoutfields.txt"
    path.write_text(f"+:h:{stream}:{','.join(fields)}\n")
    return path


def _copy_fixture(dst_path: pathlib.Path, fields, src_path: pathlib.Path, rename=None) -> pathlib.Path:
    """Copy of a wrfout fixture file with only `Times` and `fields` (optionally renamed, `{old: new}`)."""
    rename = rename or {}
    with netCDF4.Dataset(src_path) as src, netCDF4.Dataset(dst_path, "w") as dst:
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        for dim_name, dim in src.dimensions.items():
            dst.createDimension(dim_name, None if dim.isunlimited() else len(dim))
        for v in {"Times", *fields}:
            out = dst.createVariable(rename.get(v, v), src[v].dtype, src[v].dimensions)
            out.setncatts({a: src[v].getncattr(a) for a in src[v].ncattrs()})
            out[:] = src[v][:]
    return dst_path


def _simulation(fixture_dir: pathlib.Path = WRFOUT_DIR) -> Simulation:
    """Simulation over a fixture dir, beginning with its first file (so no file counts as warmup)."""
    begin = datetime.datetime.strptime(_first_file(fixture_dir).name[15:34], "%Y-%m-%d_%H:%M:%S")
    return Simulation(
        sim_dir=fixture_dir.parent, settings={}, warmup_h=12, begin=begin, end=begin + datetime.timedelta(days=1)
    )


def _run(work_dir: pathlib.Path, fixture_dir: pathlib.Path = WRFOUT_DIR, stage_cls=Cn2PostProcStage, **kwargs):
    """Run a postproc stage over a fixture into `work_dir` (uncompressed unless `compression=True`, otherwise the
    stage class's default store); return the stage and its outputs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    store = stage_cls.model_fields["store"].get_default(call_default_factory=True)
    store = store.model_copy(update={"compression": kwargs.pop("compression", False)})
    pp = stage_cls(work_dir=work_dir, wrfout_dir=fixture_dir.name, domain=1, store=store, **kwargs)
    s = _simulation(fixture_dir)
    pp.run(s)
    return pp, pp.store.outputs(pp, s)


# --- Inputs and reading --------------------------------------------------------------------------------------------


def test_discard_warmup(tmp_path):
    """Files from the warmup are dropped, but never the one containing the simulation begin, nor any file twice."""
    cases = [  # begin, first file, number of 6-hourly files, expected kept file start times
        ("2025-01-02", "2025-01-01T00", 8, ["2025-01-02_00", "2025-01-02_06", "2025-01-02_12", "2025-01-02_18"]),
        ("2025-01-01T09", "2025-01-01T00", 4, ["2025-01-01_06", "2025-01-01_12", "2025-01-01_18"]),  # 06 has begin
        ("2025-01-01", "2025-01-01T00", 3, ["2025-01-01_00", "2025-01-01_06", "2025-01-01_12"]),  # no warmup files
        ("2025-06-01", "2025-01-01T00", 3, ["2025-01-01_12"]),  # all before begin: fall back to the last one
    ]
    for i, (begin, first_file, n_files, expected) in enumerate(cases):
        wrfout_dir = tmp_path / f"case_{i}" / "wrfout"
        wrfout_dir.mkdir(parents=True)
        t0 = datetime.datetime.fromisoformat(first_file)
        for j in range(n_files):
            (wrfout_dir / f"wrfout_aux_d01_{t0 + datetime.timedelta(hours=6 * j):%Y-%m-%d_%H:%M:%S}.nc").touch()
        s = Simulation(sim_dir=wrfout_dir.parent, settings={}, warmup_h=12, begin=begin, end="2025-12-31")

        names = [
            f.name for f in PostProcStage(**_STAGE_KWARGS | {"wrfout_dir": "wrfout"}, extract_vars=[]).get_inputs(s)
        ]
        assert names == [f"wrfout_aux_d01_{t}:00:00.nc" for t in expected], f"begin={begin}"


def test_load_wrfout_from_reduced_files(tmp_path):
    """Every derived variable can be read from a file holding only its declared WRF fields, WRF fields are found by
    their ALL-CAPS name even if the file spells them lower-case, and no fill-value attributes survive.

    Guards the hand-written `requires` in `DERIVED_VARS` against mistakes and changes in wrf-python. Each entry is
    checked against the first fixture that has all its fields; an entry no fixture can verify fails.
    """
    failures = {}
    for name, derived in DERIVED_VARS.items():
        fields = [*REQUIRED_WRF_FIELDS, *derived.requires]
        fixtures = [d for d in WRFOUT_DIRS if set(fields) <= set(_fixture_fields(d))]
        if not fixtures:
            failures[name] = f"no fixture has all of {fields}"
            continue
        try:
            load_wrfout(_copy_fixture(tmp_path / f"{name}.nc", fields, _first_file(fixtures[0])), [name])
        except Exception as e:
            failures[name] = repr(e)
    assert not failures, failures

    path = _copy_fixture(tmp_path / "lower.nc", [*REQUIRED_WRF_FIELDS, "TSQ"], _first_file(), rename={"TSQ": "tsq"})
    assert "TSQ" in load_wrfout(path, ["TSQ"]).data_vars

    # wrf-python stamps _FillValue/missing_value on some vars (e.g. p), which must not survive into the dataset
    ds = load_wrfout(_first_file(), ["p", "tk", "PH"])
    for v in ds.variables:
        assert "_FillValue" not in ds[v].attrs and "missing_value" not in ds[v].attrs, v


# --- Stage configuration -------------------------------------------------------------------------------------------


def test_variable_resolution():
    """Defaults are added (also to subclass class defaults, so they must be model validators), and everything needed
    that no postproc fn returns is read from the file: postproc fn inputs come and go with the fns."""

    def to_read(pp):
        return {var_name(v) for v in pp.vars_to_read()}

    class _MinimalStage(PostProcStage):
        extract_vars: TVarList = ["th"]  # missing z/HGT/p

    for pp in [_MinimalStage(**_STAGE_KWARGS), PostProcStage(**_STAGE_KWARGS, extract_vars=["u_met"])]:
        assert {"z", "HGT", "p"} <= set(pp.extract_vars)
    assert "v_met" in pp.extract_vars  # full wind vector

    assert {"TSQ", "EL_PBL", "p", "tk"} <= to_read(Cn2PostProcStage(**_STAGE_KWARGS))  # what fn_ct2_cn2 requires
    assert not {"TSQ", "tk"} & to_read(Cn2PostProcStage(**_STAGE_KWARGS, postproc_fns=[]))
    assert not {"a", "b"} & to_read(PostProcStage(**_STAGE_KWARGS, extract_vars=["b"], postproc_fns=[_FN_A, _FN_B]))


def test_invalid_config_fails_at_init():
    """Parallel runs without resources, unknown derived variables, a field needed both raw and destaggered, and
    postproc fns in the wrong order."""
    with pytest.raises(pydantic.ValidationError, match="run_parallel"):
        PostProcStage(**_STAGE_KWARGS, extract_vars=[], run_parallel=True)

    class _ParallelByDefault(PostProcStage):  # a class default is not seen by a field validator
        run_parallel: bool = True

    with pytest.raises(pydantic.ValidationError, match="run_parallel"):
        _ParallelByDefault(**_STAGE_KWARGS, extract_vars=[])

    with pytest.raises(pydantic.ValidationError, match="exceeds the allocated CPUs"):
        PostProcStage(**_STAGE_KWARGS, extract_vars=[], run_parallel=True, resources=_RESOURCES, n_workers=3)
    with pytest.raises(pydantic.ValidationError, match="would be ignored"):
        PostProcStage(**_STAGE_KWARGS, extract_vars=[], resources=_RESOURCES, n_workers=1)

    with pytest.raises(pydantic.ValidationError, match="Unknown derived variable 'not_a_derived_var'"):
        PostProcStage(**_STAGE_KWARGS, extract_vars=["not_a_derived_var"])

    with pytest.raises(pydantic.ValidationError, match="'EL_PBL' is needed both as"):
        Cn2PostProcStage(**_STAGE_KWARGS, extract_vars=["EL_PBL"])  # fn_ct2_cn2 requires it destaggered

    with pytest.raises(pydantic.ValidationError, match="requires \\['a'\\], returned only by a later postproc fn"):
        PostProcStage(**_STAGE_KWARGS, extract_vars=["b"], postproc_fns=[_FN_B, _FN_A])


# --- Check against myoutfields.txt ---------------------------------------------------------------------------------


def test_parse_myoutfields(tmp_path):
    """Parsed like WRF: case-insensitive, per stream, `-` removes, `#` lines skipped, stops at first blank line."""
    path = tmp_path / "myoutfields.txt"
    path.write_text(
        "+:h:7:U,v, qsq ,T\n"
        "# +:h:7:COMMENTED\n"
        "+:h:7:W # trailing comment hides the whole line, like in WRF\n"
        "+:h:0:ONLY_MAIN_HISTORY\n"
        "+:i:7:ONLY_INPUT\n"
        "-:h:7:T\n"
        "\n"
        "+:h:7:AFTER_BLANK_LINE\n"
    )
    assert parse_myoutfields(path, stream=7) == {"U", "V", "QSQ"}
    assert parse_myoutfields(path, stream=0) == {"ONLY_MAIN_HISTORY"}

    path.write_text("U,V,T\n")
    with pytest.raises(ValueError, match="line 1"):
        parse_myoutfields(path, stream=7)


def test_myoutfields_check(tmp_path):
    """The check passes for what WRF wrote (in any spelling, on the configured stream, with the run's own iofields
    file) and names each missing field with the derived variable that needs it."""
    for fixture_dir in WRFOUT_DIRS:
        kwargs = dict(_STAGE_KWARGS, extract_vars=_readable_vars(fixture_dir))
        fields = _fixture_fields(fixture_dir)
        Cn2PostProcStage(**kwargs, myoutfields_path=_write_myoutfields(tmp_path, [f.lower() for f in fields]))
        Cn2PostProcStage(**kwargs, myoutfields_path=_write_myoutfields(tmp_path, fields, stream=5), iofields_stream=5)
        if (fixture_dir / "myoutfields.txt").exists():
            Cn2PostProcStage(**kwargs, myoutfields_path=fixture_dir / "myoutfields.txt")

    with pytest.raises(pydantic.ValidationError, match="history stream 7"):
        Cn2PostProcStage(**_STAGE_KWARGS, myoutfields_path=_write_myoutfields(tmp_path, _fixture_fields(), stream=5))

    fields = [f for f in _fixture_fields() if f not in ("QVAPOR", "TSQ", "XLAT")]
    with pytest.raises(pydantic.ValidationError) as excinfo:
        Cn2PostProcStage(**_STAGE_KWARGS, extract_vars=["rh"], myoutfields_path=_write_myoutfields(tmp_path, fields))
    msg = str(excinfo.value)
    assert "QVAPOR (for rh)" in msg  # needed by a derived variable only
    assert re.search(r"\bTSQ\b(?! \()", msg)  # a WRF field read as it is (for fn_ct2_cn2), so nothing to add
    assert "XLAT (for coordinates)" in msg  # needed by wrf-python for every variable


# --- Runs ----------------------------------------------------------------------------------------------------------


def test_stage_fully_executed(tmp_path):
    """A Cn2 stage reading everything a fixture provides (plus the water paths where rain and snow are written) runs
    through every input file and stores exactly its `extract_vars` and what its postproc fns return, complete and
    finite."""
    for fixture_dir in WRFOUT_DIRS:
        fields = _fixture_fields(fixture_dir)
        postproc_fns = [fn_ct2_cn2] + ([fn_int_water_paths] if {"QRAIN", "QSNOW"} <= set(fields) else [])
        pp = Cn2PostProcStage(
            work_dir=tmp_path / fixture_dir.name,
            wrfout_dir=fixture_dir.name,
            domain=1,
            store=NetCDFStore(file_suffix="cn2", compression=False),
            extract_vars=_readable_vars(fixture_dir),
            postproc_fns=postproc_fns,
            myoutfields_path=_write_myoutfields(tmp_path, fields),
        )
        s = _simulation(fixture_dir)
        pp.work_dir.mkdir()  # absolute work dirs are not created by the stage
        pp.setup(s)
        assert pp.is_setup(s) and not pp.is_done(s)
        pp.run(s)
        assert pp.is_done(s)

        inputs, outputs = pp.get_inputs(s), pp.store.outputs(pp, s)
        assert len(outputs) == len(inputs) == 2

        expected = {var_name(v) for v in pp.extract_vars} | {name for fn in postproc_fns for name in fn.returns}
        for f_in, f_out in zip(inputs, outputs):
            ds = xr.open_dataset(f_out)
            stored = set(ds.data_vars) | set(ds.coords)  # requested XLAT/XLONG end up as the coordinates they are
            extra = set(ds.data_vars) - expected
            assert expected <= stored and not extra, f"{f_out.name}: missing {expected - stored}, extra {extra}"
            with netCDF4.Dataset(f_in) as wrfin:
                assert ds.sizes["time"] == len(wrfin.dimensions["Time"])  # all time steps of the input file
            not_finite = [v for v in ds.data_vars if not np.all(np.isfinite(ds[v].values))]
            assert not not_finite, f"{f_out.name}: non-finite values in {not_finite}"


def test_parallel_run(tmp_path):
    """Parallel workers produce the same outputs as a serial run; uvmet is split without its stacking dimension."""
    kwargs = dict(extract_vars=["w", "u_met", "v_met"], resources=_RESOURCES)
    _, serial = _run(tmp_path / "serial", **kwargs)
    _, parallel = _run(tmp_path / "parallel", run_parallel=True, **kwargs)

    assert [f.name for f in serial] == [f.name for f in parallel]
    for f_s, f_p in zip(serial, parallel):
        ds = xr.open_dataset(f_p)
        assert ds.identical(xr.open_dataset(f_s))
        assert {"w", "u_met", "v_met", "ct2", "cn2"} <= set(ds.data_vars) and "u_v" not in ds.dims


def test_framewise_output_matches_whole_file(tmp_path):
    """Files are processed and written one frame at a time; the result equals processing all frames at once."""
    kwargs = dict(extract_vars=["w", "u_met", "v_met", "p", "HGT", ("EL_PBL", "bottom_top_stag")], resources=_RESOURCES)
    for compression in (False, True):
        pp, outputs = _run(tmp_path / str(compression), compression=compression, **kwargs)
        inputs = pp.get_inputs(_simulation(WRFOUT_DIR))
        assert len(outputs) == len(inputs) > 0
        for f_in, f_out in zip(inputs, outputs):
            whole = pp._process_frame(f_in, None)
            assert whole.sizes["time"] > 1  # the fixtures must exercise appending
            f_whole = tmp_path / f"whole_{f_out.name}"  # written in one go, as before frame-wise processing
            whole.to_netcdf(f_whole, engine="h5netcdf")
            xr.testing.assert_identical(xr.open_dataset(f_out), xr.open_dataset(f_whole))
            assert not list(f_out.parent.glob(".*.tmp"))  # temp file renamed


def test_n_workers(tmp_path, monkeypatch):
    """`n_workers` caps the worker pool below the allocated CPUs; unset, all allocated CPUs are used."""
    import concurrent.futures

    seen = []
    pool = concurrent.futures.ProcessPoolExecutor

    def recording_pool(max_workers=None, **kwargs):
        seen.append(max_workers)
        return pool(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", recording_pool)
    kwargs = dict(extract_vars=["w"], run_parallel=True, resources=_RESOURCES)
    _run(tmp_path / "all", **kwargs)
    _, outputs = _run(tmp_path / "capped", n_workers=1, **kwargs)
    assert seen == [_RESOURCES.cpus_total, 1]
    assert len(outputs) == 2


def test_written_outputs(tmp_path):
    """Outputs are float32 (wrf-python returns z/w/u_met/v_met/slp as float64) and read back without fill-value
    warnings."""
    _, (f_single, *_) = _run(tmp_path, extract_vars=["w", "u_met", "p", "tk", "slp"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = xr.open_dataset(f_single).load()
    serialization = [w for w in caught if issubclass(w.category, xr.SerializationWarning)]
    assert not serialization, [str(w.message) for w in serialization]

    assert not [v for v in ds.data_vars if ds[v].dtype == np.float64]
    wrf_dtypes = load_wrfout(_first_file(WRFOUT_DIR), ["z", "w", "u_met", "slp", "tk"], timeidx=0)
    assert all(wrf_dtypes[v].dtype == np.float32 for v in wrf_dtypes.data_vars)  # cast right after reading

    # Cn2 spans very small magnitudes (1e-22..1e-13), far above the float32 subnormal limit -> no underflow
    assert ds["cn2"].dtype == np.float32
    assert np.all(np.isfinite(ds["cn2"].values)) and ds["cn2"].max() > 0


def test_postproc_fn_replaces_derived_variable(tmp_path):
    """A postproc fn returning a derived variable's name replaces the wrf-python one; its WRF fields are read."""
    my_p = PostProcFn(fn=lambda ds: {"p": (ds["P"] + ds["PB"]) * 0 + 1}, requires=["P", "PB"], returns=["p"])
    kwargs = dict(extract_vars=["p"], postproc_fns=[my_p], discard_warmup=False)  # keep all files, warmup or not
    pp, (f_out, *_) = _run(tmp_path, stage_cls=PostProcStage, **kwargs)
    assert "p" not in pp.vars_to_read() and {"P", "PB"} <= set(pp.vars_to_read())

    ds = xr.open_dataset(f_out)
    assert np.all(ds["p"] == 1)
    assert "P" not in ds.data_vars  # inputs of postproc fns are not stored


def test_failure_names_input_file(tmp_path):
    """A failing file must be identifiable from the error, also when it fails inside a worker process."""
    for run_parallel in (False, True):
        with pytest.raises(RuntimeError) as excinfo:
            _run(tmp_path, extract_vars=["NOT_A_WRF_VARIABLE"], run_parallel=run_parallel, resources=_RESOURCES)
        msg = str(excinfo.value) + str(excinfo.value.__cause__)
        assert any(f.name in msg for f in WRFOUT_DIR.glob("wrfout_aux_d01_*")), msg  # workers finish in any order


def test_crop(tmp_path):
    """`crop` cuts the outer cells of the horizontal grid after processing: the output equals the full output's
    `[n:-n]` in y and x (derived fields computed on the full grid), coordinates included; too large a crop fails."""
    kwargs = dict(extract_vars=["w", "u_met", "th", "HGT"])
    _, full = _run(tmp_path / "full", **kwargs)
    pp, cropped = _run(tmp_path / "cropped", crop=1, **kwargs)
    for f_full, f_crop in zip(full, cropped):
        a, b = xr.open_dataset(f_full), xr.open_dataset(f_crop)
        assert b.attrs["crop"] == 1 and "crop" not in a.attrs
        expected = a.isel(south_north=slice(1, -1), west_east=slice(1, -1))
        xr.testing.assert_identical(b.drop_attrs(deep=False), expected.drop_attrs(deep=False))
    with pytest.raises(RuntimeError, match="leaves no grid points"):
        _run(tmp_path / "too_much", crop=3, **kwargs)  # 5 x 5 fixture
