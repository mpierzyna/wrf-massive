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
from fixtures import WRFOUT_DIR, WRFOUT_DIRS, simple_simulation

pytest.importorskip("wrf", reason="wrf-python is not installed (`postproc` extra)")


from wrf_massive.base import Resources, Simulation
from wrf_massive.stages.postproc import PostProcFn, PostProcStage
from wrf_massive.stages.postproc.base import TVarList
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage, fn_ct2_cn2
from wrf_massive.stages.postproc.iofields import parse_myoutfields
from wrf_massive.stages.postproc.precip import fn_int_water_paths
from wrf_massive.stages.postproc.variables import DERIVED_VARS, REQUIRED_WRF_FIELDS, var_name, wrf_fields_of
from wrf_massive.stages.postproc.wrfout import load_wrfout


def _make_warmup_stage(tmp_path, begin: str, first_file: str, n_files: int = 6, freq_h: int = 6) -> tuple:
    """Build a stage with `n_files` empty wrfout files, `freq_h` apart, starting at `first_file`."""
    sim_dir = tmp_path / "sim"
    wrfout_dir = sim_dir / "wrfout"
    wrfout_dir.mkdir(parents=True)

    t0 = datetime.datetime.fromisoformat(first_file)
    for i in range(n_files):
        t = t0 + datetime.timedelta(hours=freq_h * i)
        (wrfout_dir / f"wrfout_aux_d01_{t:%Y-%m-%d_%H:%M:%S}.nc").touch()

    s = Simulation(sim_dir=sim_dir, settings={}, warmup_h=12, begin=begin, end="2025-12-31")
    pp = PostProcStage(
        work_dir=tmp_path / "out",
        wrfout_dir="wrfout",
        domain=1,
        extract_vars=[],
        compression=False,
        resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
    )
    return pp, s


def test_discard_warmup_no_duplicates(tmp_path):
    """Every kept input must appear exactly once (parallel workers would otherwise clash on the same output)."""
    pp, s = _make_warmup_stage(tmp_path, begin="2025-01-02", first_file="2025-01-01T00:00:00", n_files=8)
    inputs = pp.get_inputs(s)
    assert len(inputs) == len(set(inputs))


def test_discard_warmup_keeps_files_at_and_after_begin(tmp_path):
    """File starting exactly at begin exists -> keep it and everything after, drop all warmup files."""
    pp, s = _make_warmup_stage(tmp_path, begin="2025-01-02", first_file="2025-01-01T00:00:00", n_files=8)
    names = [f.name for f in pp.get_inputs(s)]
    assert names == [
        "wrfout_aux_d01_2025-01-02_00:00:00.nc",
        "wrfout_aux_d01_2025-01-02_06:00:00.nc",
        "wrfout_aux_d01_2025-01-02_12:00:00.nc",
        "wrfout_aux_d01_2025-01-02_18:00:00.nc",
    ]


def test_discard_warmup_keeps_file_containing_begin(tmp_path):
    """No file starts exactly at begin -> keep the last file before it, since it contains the simulation start."""
    pp, s = _make_warmup_stage(tmp_path, begin="2025-01-01T09:00:00", first_file="2025-01-01T00:00:00", n_files=4)
    names = [f.name for f in pp.get_inputs(s)]
    assert names == [
        "wrfout_aux_d01_2025-01-01_06:00:00.nc",  # contains begin
        "wrfout_aux_d01_2025-01-01_12:00:00.nc",
        "wrfout_aux_d01_2025-01-01_18:00:00.nc",
    ]


def test_discard_warmup_without_warmup_files(tmp_path):
    """No file before begin -> nothing may be dropped (regression: first file used to be skipped)."""
    pp, s = _make_warmup_stage(tmp_path, begin="2025-01-01", first_file="2025-01-01T00:00:00", n_files=3)
    assert len(pp.get_inputs(s)) == 3


def test_discard_warmup_all_files_before_begin(tmp_path):
    """All files start before begin -> fall back to the last one."""
    pp, s = _make_warmup_stage(tmp_path, begin="2025-06-01", first_file="2025-01-01T00:00:00", n_files=3)
    names = [f.name for f in pp.get_inputs(s)]
    assert names == ["wrfout_aux_d01_2025-01-01_12:00:00.nc"]


@pytest.mark.parametrize("run_parallel", [False, True])
def test_w(simple_simulation: Simulation, tmp_path, run_parallel):
    """Test that wa (wrfout) gets renamed to w using postprocessing functions."""
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=WRFOUT_DIR,
        domain=1,
        extract_vars=["w"],
        compression=False,  # for testing
        run_parallel=run_parallel,
        discard_warmup=False,  # for testing
        resources=Resources(n_tasks=2, cpus_per_task=1, mem_per_cpu="1G"),
    )

    # w is a wrf-python derived variable (`wa`), read from the file without any postproc fn
    assert pp.postproc_fns == []
    assert "w" in pp.vars_to_read()

    # Run stage
    pp.run(simple_simulation)

    # Check that saved dataset has w
    ds = xr.open_mfdataset(list(sorted(pp.work_dir.glob(f"*{pp.file_suffix}.nc"))))
    assert "w" in ds.data_vars


@pytest.mark.parametrize("run_parallel", [False, True])
def test_uv(simple_simulation: Simulation, tmp_path, run_parallel):
    """Test that uvmet variable (wrfout) gets split into individual u_met and v_met."""
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=WRFOUT_DIR,
        domain=1,
        extract_vars=["u_met", "v_met"],
        run_parallel=run_parallel,
        compression=False,  # for testing
        discard_warmup=False,  # for testing
        resources=Resources(n_tasks=2, cpus_per_task=1, mem_per_cpu="1G"),
    )

    # u_met and v_met are the components of the wrf-python derived variable `uvmet`
    assert {"u_met", "v_met"} <= {str(v) for v in pp.vars_to_read()}

    # Run stage
    pp.run(simple_simulation)

    # Check that saved dataset has the calculated variables
    ds = xr.open_mfdataset(list(sorted(pp.work_dir.glob(f"*{pp.file_suffix}.nc"))))
    assert "u_met" in ds.data_vars
    assert "v_met" in ds.data_vars
    assert "u_v" not in ds.dims


@pytest.mark.parametrize("run_parallel", [False, True])
def test_cn2(simple_simulation: Simulation, tmp_path, run_parallel):
    """Test that CT2 and Cn2 variables are calculated correctly with injected dependencies."""
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=WRFOUT_DIR,
        domain=1,
        extract_vars=[],  # leave this empty, work only with dependencies
        postproc_fns=[fn_ct2_cn2],
        compression=False,  # for testing
        run_parallel=run_parallel,  # for testing
        discard_warmup=False,  # for testing
        resources=Resources(n_tasks=2, cpus_per_task=1, mem_per_cpu="1G"),
    )

    # Run stage
    pp.run(simple_simulation)

    # Check that saved dataset has the calculated variables
    ds = xr.open_mfdataset(list(sorted(pp.work_dir.glob(f"*{pp.file_suffix}.nc"))))
    assert "cn2" in ds.data_vars
    assert "ct2" in ds.data_vars


# --- Helpers over the wrfout fixtures -------------------------------------------------------------------------------
# The fixtures differ in which fields WRF wrote (see fixtures.py). Tests take what they need from the fixtures'
# contents instead of naming a particular setup, so they hold for any WRF output configuration.

_STAGE_KWARGS = dict(work_dir="/tmp", wrfout_dir="x", domain=1, compression=False)


def _first_file(fixture_dir: pathlib.Path = WRFOUT_DIR) -> pathlib.Path:
    return sorted(fixture_dir.glob("wrfout_aux_d01_*"))[0]


def _fixture_fields(fixture_dir: pathlib.Path = WRFOUT_DIR) -> list:
    """WRF fields in a wrfout fixture (what the WRF run behind it wrote to its aux stream)."""
    with netCDF4.Dataset(_first_file(fixture_dir)) as f:
        return [v for v in f.variables if v != "Times"]


def _fixtures_with(fields) -> list:
    """Fixture dirs whose files contain all of `fields`."""
    return [d for d in WRFOUT_DIRS if set(fields) <= set(_fixture_fields(d))]


def _readable_vars(fixture_dir: pathlib.Path) -> TVarList:
    """Everything a fixture provides: the WRF fields and derived variables it has all needed fields for.

    EL_PBL is requested destaggered, as `fn_ct2_cn2` needs it. The horizontally staggered U/V are left out: wrf-python
    reads them raw only with their own coordinates (XLAT_U/XLONG_U, XLAT_V/XLONG_V), which wrfout_aux files lack.
    Their earth-relative versions `u_met`/`v_met` are included as derived variables.
    """
    fields = set(_fixture_fields(fixture_dir)) - {"U", "V"}
    names = [*sorted(fields), *DERIVED_VARS]
    readable = [n for n in names if set(wrf_fields_of(n)) <= set(_fixture_fields(fixture_dir))]
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


def _simulation(fixture_dir: pathlib.Path) -> Simulation:
    """Simulation over a fixture dir, beginning with its first file (so no file counts as warmup)."""
    begin = datetime.datetime.strptime(_first_file(fixture_dir).name[15:34], "%Y-%m-%d_%H:%M:%S")
    return Simulation(
        sim_dir=fixture_dir.parent, settings={}, warmup_h=12, begin=begin, end=begin + datetime.timedelta(days=1)
    )


def _make_cn2_stage(tmp_path, fixture_dir: pathlib.Path = WRFOUT_DIR, **kwargs) -> tuple:
    """Cn2 stage over a wrfout fixture, plus a matching simulation."""
    pp = Cn2PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=fixture_dir.name,
        domain=1,
        compression=False,  # for testing
        resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
        **kwargs,
    )
    return pp, _simulation(fixture_dir)


# --- Reading and writing -------------------------------------------------------------------------------------------


def test_fill_values_normalized():
    """wrf-python stamps _FillValue/missing_value on some vars, which must not survive into the dataset."""
    ds = load_wrfout(_first_file(), ["p", "tk", "PH"])
    for v in ds.variables:
        assert "_FillValue" not in ds[v].attrs, f"{v} still carries _FillValue"
        assert "missing_value" not in ds[v].attrs, f"{v} still carries missing_value"


def test_written_file_reads_without_warning(tmp_path):
    """Conflicting fill values used to raise a SerializationWarning on every read of the output."""
    pp, s = _make_cn2_stage(tmp_path, extract_vars=["p", "tk", "slp"])
    pp.run(s)

    f_out = sorted(tmp_path.glob(f"*{pp.file_suffix}.nc"))[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xr.open_dataset(f_out).load()

    serialization = [w for w in caught if issubclass(w.category, xr.SerializationWarning)]
    assert not serialization, [str(w.message) for w in serialization]


@pytest.mark.parametrize("single_precision", [True, False])
def test_single_precision(tmp_path, single_precision):
    """wrf-python returns z/w/u_met/v_met/slp as float64; they should be stored as float32 by default."""
    pp, s = _make_cn2_stage(tmp_path, extract_vars=["w", "u_met", "slp"], single_precision=single_precision)
    pp.run(s)

    ds = xr.open_dataset(sorted(tmp_path.glob(f"*{pp.file_suffix}.nc"))[0])
    f64 = sorted(v for v in ds.data_vars if ds[v].dtype == np.float64)
    if single_precision:
        assert not f64, f"expected no float64 data vars, got {f64}"
    else:
        assert "z" in f64 and "w" in f64  # unchanged wrf-python dtypes

    # Cn2 spans very small magnitudes (1e-22..1e-13), far above the float32 subnormal limit -> no underflow
    assert ds["cn2"].dtype == np.float32
    cn2 = ds["cn2"].values
    assert np.all(np.isfinite(cn2))
    assert cn2.max() > 0


def test_wrf_fields_read_case_insensitively(tmp_path):
    """WRF writes upper-case names, but a lower-case field in the file is still found by its ALL-CAPS name."""
    path = _copy_fixture(tmp_path / "lower.nc", [*REQUIRED_WRF_FIELDS, "TSQ"], _first_file(), rename={"TSQ": "tsq"})
    assert "TSQ" in load_wrfout(path, ["TSQ"]).data_vars


@pytest.mark.parametrize("name", list(DERIVED_VARS))
def test_derived_requires_are_sufficient(tmp_path, name):
    """Every derived variable can be read from a wrfout file holding only its declared WRF fields.

    Guards the hand-written `requires` in `DERIVED_VARS` against mistakes and against changes in wrf-python. Each entry
    is checked against the first fixture that has all its fields; an entry no fixture can verify fails.
    """
    fields = [*REQUIRED_WRF_FIELDS, *DERIVED_VARS[name].requires]
    fixtures = _fixtures_with(fields)
    assert fixtures, f"No test fixture has all of {fields}, so '{name}' cannot be verified."

    path = _copy_fixture(tmp_path / f"{name}.nc", fields, _first_file(fixtures[0]))
    assert name in load_wrfout(path, [name]).data_vars


# --- Stage configuration -------------------------------------------------------------------------------------------


def test_parallel_requires_resources(tmp_path):
    """Parallel runs take the worker count from `resources`, so building one without must fail loudly."""
    kwargs = dict(work_dir=tmp_path, wrfout_dir="x", domain=1, extract_vars=[], compression=False)

    with pytest.raises(pydantic.ValidationError, match="run_parallel"):
        PostProcStage(**kwargs, run_parallel=True)

    PostProcStage(**kwargs)  # serial runs need no resources

    # Must also catch subclasses that turn parallelism on via a class default (a field validator would not)
    class _ParallelByDefault(PostProcStage):
        run_parallel: bool = True

    with pytest.raises(pydantic.ValidationError, match="run_parallel"):
        _ParallelByDefault(**kwargs)


@pytest.mark.parametrize("run_parallel", [False, True])
def test_failure_names_input_file(tmp_path, run_parallel):
    """A failing file must be identifiable from the error, also when it fails inside a worker process."""
    pp, s = _make_cn2_stage(tmp_path, extract_vars=["NOT_A_WRF_VARIABLE"], run_parallel=run_parallel)

    with pytest.raises(RuntimeError) as excinfo:
        pp.run(s)

    assert _first_file().name in str(excinfo.value) + str(excinfo.value.__cause__)


def test_defaults_applied_to_subclass_class_defaults():
    """`extract_vars` defaults of a subclass are never seen by field validators -> must be model validators."""

    class _MinimalStage(PostProcStage):
        extract_vars: TVarList = ["th"]  # missing z/HGT/p

    pp = _MinimalStage(work_dir="/tmp", wrfout_dir="x", domain=1, compression=False)
    for v in ["z", "HGT", "p"]:
        assert v in pp.extract_vars

    # Explicitly passed values must keep working the same way (this is what the field validator used to cover)
    pp = PostProcStage(work_dir="/tmp", wrfout_dir="x", domain=1, compression=False, extract_vars=["u_met"])
    for v in ["z", "HGT", "p"]:
        assert v in pp.extract_vars
    assert "v_met" in pp.extract_vars


def test_postproc_fn_inputs_read_without_extract_vars():
    """Everything a postproc fn requires is read from the file; it comes and goes with the postproc fns."""
    to_read = [var_name(v) for v in Cn2PostProcStage(**_STAGE_KWARGS).vars_to_read()]
    assert {"TSQ", "EL_PBL", "p", "tk"} <= set(to_read)  # what fn_ct2_cn2 requires

    to_read = [var_name(v) for v in Cn2PostProcStage(**_STAGE_KWARGS, postproc_fns=[]).vars_to_read()]
    assert "tk" not in to_read and "TSQ" not in to_read


def test_unknown_derived_variable_fails():
    with pytest.raises(pydantic.ValidationError, match="Unknown derived variable 'not_a_derived_var'"):
        PostProcStage(**_STAGE_KWARGS, extract_vars=["not_a_derived_var"])


def test_variable_read_two_ways_fails():
    """A field stored as it is but required destaggered (or vice versa) cannot be read both ways."""
    with pytest.raises(pydantic.ValidationError, match="'EL_PBL' is needed both as"):
        Cn2PostProcStage(**_STAGE_KWARGS, extract_vars=["EL_PBL"])
    Cn2PostProcStage(**_STAGE_KWARGS, extract_vars=[("EL_PBL", "bottom_top_stag")])


def test_postproc_fn_order():
    """A postproc fn sees what earlier fns return, but not what later ones return."""
    fn_a = PostProcFn(fn=lambda ds: {"a": ds["tk"] * 2}, requires=["tk"], returns=["a"])
    fn_b = PostProcFn(fn=lambda ds: {"b": ds["a"] + 1}, requires=["a"], returns=["b"])

    pp = PostProcStage(**_STAGE_KWARGS, extract_vars=["b"], postproc_fns=[fn_a, fn_b])
    assert "a" not in pp.vars_to_read() and "b" not in pp.vars_to_read()

    with pytest.raises(pydantic.ValidationError, match="requires \\['a'\\], returned only by a later postproc fn"):
        PostProcStage(**_STAGE_KWARGS, extract_vars=["b"], postproc_fns=[fn_b, fn_a])


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


@pytest.mark.parametrize("fixture_dir", WRFOUT_DIRS, ids=lambda d: d.name)
def test_myoutfields_check_passes_for_what_wrf_wrote(tmp_path, fixture_dir):
    """A stage reading everything a fixture provides passes the check against the fields the fixture holds."""
    kwargs = dict(_STAGE_KWARGS, extract_vars=_readable_vars(fixture_dir))
    Cn2PostProcStage(**kwargs, myoutfields_path=_write_myoutfields(tmp_path, _fixture_fields(fixture_dir)))

    # Case-insensitive like WRF, so lower-case entries in myoutfields.txt count as well
    lower = [f.lower() for f in _fixture_fields(fixture_dir)]
    Cn2PostProcStage(**kwargs, myoutfields_path=_write_myoutfields(tmp_path, lower))

    # The iofields file of the WRF run, if the fixture has it
    if (fixture_dir / "myoutfields.txt").exists():
        Cn2PostProcStage(**kwargs, myoutfields_path=fixture_dir / "myoutfields.txt")


def test_myoutfields_check_names_missing_fields(tmp_path):
    """A missing field fails at init, named with the derived variable that needs it (if any)."""
    fields = [f for f in _fixture_fields() if f not in ("QVAPOR", "TSQ", "XLAT")]
    path = _write_myoutfields(tmp_path, fields)

    with pytest.raises(pydantic.ValidationError) as excinfo:
        Cn2PostProcStage(**_STAGE_KWARGS, extract_vars=["rh"], myoutfields_path=path)
    msg = str(excinfo.value)
    assert "QVAPOR (for rh)" in msg  # needed by a derived variable only
    assert re.search(r"\bTSQ\b(?! \()", msg)  # a WRF field read as it is (for fn_ct2_cn2), so nothing to add
    assert "XLAT (for coordinates)" in msg  # needed by wrf-python for every variable


def test_myoutfields_check_uses_configured_stream(tmp_path):
    """Only the fields of `iofields_stream` count."""
    path = _write_myoutfields(tmp_path, _fixture_fields(), stream=5)
    with pytest.raises(pydantic.ValidationError, match="history stream 7"):
        Cn2PostProcStage(**_STAGE_KWARGS, myoutfields_path=path)
    Cn2PostProcStage(**_STAGE_KWARGS, myoutfields_path=path, iofields_stream=5)


# --- Full runs -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("run_parallel", [False, True])
@pytest.mark.parametrize("read_everything", [False, True], ids=["default", "everything"])
@pytest.mark.parametrize("fixture_dir", WRFOUT_DIRS, ids=lambda d: d.name)
def test_cn2_stage_fully_executed(tmp_path, fixture_dir, read_everything, run_parallel):
    """The Cn2 stage, checked against what WRF wrote, runs through every input file and stores exactly its
    `extract_vars` plus what its postproc fns return, complete and finite. Either with the defaults or reading
    everything the fixture provides."""
    extract_vars = _readable_vars(fixture_dir) if read_everything else []
    pp, s = _make_cn2_stage(
        tmp_path,
        fixture_dir,
        extract_vars=extract_vars,
        myoutfields_path=_write_myoutfields(tmp_path, _fixture_fields(fixture_dir)),
        run_parallel=run_parallel,
    )
    pp.setup(s)
    assert pp.is_setup(s) and not pp.is_done(s)
    pp.run(s)
    assert pp.is_done(s)

    outputs = sorted(tmp_path.glob(f"*{pp.file_suffix}.nc"))
    inputs = pp.get_inputs(s)
    assert len(outputs) == len(inputs) == 2

    expected = {var_name(v) for v in pp.extract_vars} | {"ct2", "cn2"}
    assert {"z", "HGT", "p"} <= expected  # always added by PostProcStage
    for f_in, f_out in zip(inputs, outputs):
        ds = xr.open_dataset(f_out)
        stored = set(ds.data_vars) | set(ds.coords)  # requested XLAT/XLONG end up as the coordinates they are
        extra = set(ds.data_vars) - expected
        assert expected <= stored and not extra, f"missing {expected - stored}, extra {extra}"
        with netCDF4.Dataset(f_in) as wrfin:
            assert ds.sizes["time"] == len(wrfin.dimensions["Time"])  # all time steps of the input file
        assert ds["cn2"].sizes["bottom_top"] == ds["p"].sizes["bottom_top"]  # on mass levels
        not_finite = [v for v in ds.data_vars if not np.all(np.isfinite(ds[v].values))]
        assert not not_finite, f"{f_out.name}: non-finite values in {not_finite}"


def test_water_paths_and_reflectivity(tmp_path):
    """mdbz and the water paths need rain and snow, so they run on a fixture with QRAIN/QSNOW."""
    fixtures = _fixtures_with(["QRAIN", "QSNOW"])
    assert fixtures, "No test fixture has QRAIN/QSNOW."
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=fixtures[0].name,
        domain=1,
        extract_vars=["mdbz"],
        postproc_fns=[fn_int_water_paths],
        myoutfields_path=_write_myoutfields(tmp_path, _fixture_fields(fixtures[0])),
        compression=False,
    )
    pp.run(_simulation(fixtures[0]))

    ds = xr.open_mfdataset(sorted(tmp_path.glob(f"*{pp.file_suffix}.nc")))
    assert set(ds.data_vars) == {"mdbz", "z", "HGT", "p", "rwp", "swp", "iwv"}
    assert all(np.all(np.isfinite(ds[v].values)) for v in ds.data_vars)


def test_postproc_fn_replaces_derived_variable(simple_simulation: Simulation, tmp_path):
    """A postproc fn returning a derived variable's name replaces the wrf-python one; its WRF fields are read."""
    my_p = PostProcFn(fn=lambda ds: {"p": (ds["P"] + ds["PB"]) * 0 + 1}, requires=["P", "PB"], returns=["p"])
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=WRFOUT_DIR,
        domain=1,
        extract_vars=["p"],
        postproc_fns=[my_p],
        compression=False,
        discard_warmup=False,
    )
    assert "p" not in pp.vars_to_read()
    pp.run(simple_simulation)

    ds = xr.open_dataset(sorted(tmp_path.glob(f"*{pp.file_suffix}.nc"))[0])
    assert np.all(ds["p"] == 1)
    assert "P" not in ds.data_vars and "tk" not in ds.data_vars  # inputs of postproc fns are not stored
