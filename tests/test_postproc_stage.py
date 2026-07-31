import datetime

import pytest
import xarray as xr
from fixtures import WRFOUT_DIR, simple_simulation

from wrf_massive.base import Resources, Simulation
from wrf_massive.stages.postproc import PostProcStage
from wrf_massive.stages.postproc.cn2 import fn_ct2_cn2


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

    # Check that w gets moved to postproc functions
    assert "w" not in pp.extract_vars
    assert "w" in pp.postproc_fns[0].returns

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

    # Check that u_met and v_met get moved to postproc functions
    assert ("u_met" not in pp.extract_vars) and ("v_met" not in pp.extract_vars)
    assert "uvmet" in pp.postproc_fns[0].requires
    assert "u_met" in pp.postproc_fns[0].returns
    assert "v_met" in pp.postproc_fns[0].returns

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
