import pytest
import xarray as xr
from fixtures import WRFOUT_DIR, simple_simulation

from wrf_massive.base import Resources, Simulation
from wrf_massive.stages.postproc import PostProcStage
from wrf_massive.stages.postproc.cn2 import fn_ct2_cn2


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
