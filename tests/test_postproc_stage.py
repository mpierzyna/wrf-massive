import pytest
import xarray as xr
from fixtures import WRFOUT_DIR, simple_simulation

from wrf_massive.base import Simulation
from wrf_massive.stages.postproc import PostProcStage
from wrf_massive.stages.postproc.cn2 import get_ct2_cn2_fn


def test_w(simple_simulation: Simulation, tmp_path):
    """Test that wa (wrfout) gets renamed to w using postprocessing functions."""
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=WRFOUT_DIR,
        domain=1,
        extract_vars=["w"],
        compression=False,  # for testing
        run_parallel=False,  # for testing
        discard_warmup=False,  # for testing
    )

    # Check that w gets moved to postproc functions
    assert "w" not in pp.extract_vars
    assert "w" in pp.postproc_fns[0].returns

    # Run stage
    pp.run(simple_simulation)

    # Check that saved dataset has w
    ds = xr.open_mfdataset(list(sorted(pp.work_dir.glob(f"*{pp.file_suffix}.nc"))))
    assert "w" in ds.data_vars


def test_uv(simple_simulation: Simulation, tmp_path):
    """Test that uvmet variable (wrfout) gets split into individual u_met and v_met."""
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=WRFOUT_DIR,
        domain=1,
        extract_vars=["u_met", "v_met"],
        compression=False,  # for testing
        run_parallel=False,  # for testing
        discard_warmup=False,  # for testing
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


def test_cn2(simple_simulation: Simulation, tmp_path):
    """Test that CT2 and Cn2 variables are calculated correctly with injected dependencies."""
    pp = PostProcStage(
        work_dir=tmp_path,
        wrfout_dir=WRFOUT_DIR,
        domain=1,
        extract_vars=[],  # leave this empty, work only with dependencies
        postproc_fns=[get_ct2_cn2_fn()],
        compression=False,  # for testing
        run_parallel=False,  # for testing
        discard_warmup=False,  # for testing
    )

    # Run stage
    pp.run(simple_simulation)

    # Check that saved dataset has the calculated variables
    ds = xr.open_mfdataset(list(sorted(pp.work_dir.glob(f"*{pp.file_suffix}.nc"))))
    assert "cn2" in ds.data_vars
    assert "ct2" in ds.data_vars
