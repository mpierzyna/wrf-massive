# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""NetCDFStore writer semantics on synthetic frames (no wrf-python needed)."""

import pathlib

import numpy as np
import pandas as pd
import pydantic
import pytest
import xarray as xr

from wrf_massive.store.netcdf import NetCDFStore

INPUT = pathlib.Path("wrfout_aux_d01_2025-01-01_00:00:00")
N = 4


class _Stage:
    """Stand-in for a postproc stage: the store only needs its work dir."""

    def __init__(self, work_dir: pathlib.Path):
        self.work_dir = work_dir

    def get_work_dir(self, s) -> pathlib.Path:
        return self.work_dir


def _frames(n: int = N) -> xr.Dataset:
    rng = np.random.default_rng(0)
    time = pd.date_range("2025-01-01", periods=n, freq="h")
    return xr.Dataset(
        {
            "a": (("time", "bottom_top", "south_north", "west_east"), rng.random((n, 3, 4, 5), dtype=np.float32)),
            "b": (("time", "south_north", "west_east"), rng.random((n, 4, 5), dtype=np.float32)),
            "static": (("south_north", "west_east"), rng.random((4, 5), dtype=np.float32)),
        },
        coords={"time": time},
    )


def _write(store, stage, ds, chunks, fail_at=None):
    """Write `ds` through the store's writer in consecutive pieces of the given sizes."""
    with store.writer(stage, None, INPUT) as w:
        assert w.pending_frames(ds.sizes["time"]) == list(range(ds.sizes["time"]))
        i0 = 0
        for n in chunks:
            if fail_at == i0:
                raise RuntimeError("processing failed")
            w.write(ds.isel(time=slice(i0, i0 + n)), i0)
            i0 += n


@pytest.mark.parametrize("compression", [False, True])
def test_framewise_and_multiframe_writes_give_the_same_file(tmp_path, compression):
    """Single frames and several frames per write (as ingest does) give the dataset back, temp file removed."""
    ds = _frames()
    store = NetCDFStore(compression=compression, file_suffix="x")
    for chunks in ([1, 1, 1, 1], [2, 2], [1, 3], [N]):
        stage = _Stage(tmp_path / "-".join(map(str, chunks)))
        stage.work_dir.mkdir()
        _write(store, stage, ds, chunks)

        (f_out,) = store.outputs(stage, None)
        assert f_out.name == "wrfout_aux_d01_2025-01-01_00-00-00_x.nc"
        xr.testing.assert_identical(xr.open_dataset(f_out).load(), ds)
        assert not list(stage.work_dir.glob(".*"))  # temp file renamed
        assert store.is_done(stage, None, [INPUT])


def test_failure_leaves_no_output(tmp_path):
    """A failure mid-file leaves neither an output nor a temp file, so a re-run starts the file again."""
    store, stage = NetCDFStore(), _Stage(tmp_path)
    with pytest.raises(RuntimeError, match="processing failed"):
        _write(store, stage, _frames(), [1, 1, 1, 1], fail_at=2)
    assert not list(tmp_path.iterdir())
    assert not store.is_done(stage, None, [INPUT])


def test_incomplete_or_unordered_writes_raise(tmp_path):
    """Leaving the writer with frames missing, or writing frames out of order, is an error and writes nothing."""
    store, stage, ds = NetCDFStore(), _Stage(tmp_path), _frames()
    with pytest.raises(RuntimeError, match="only 2 of 4 frames"), store.writer(stage, None, INPUT) as w:
        w.pending_frames(N)
        w.write(ds.isel(time=slice(0, 2)), 0)
    with pytest.raises(ValueError, match="expected frame 0, got 1"), store.writer(stage, None, INPUT) as w:
        w.pending_frames(N)
        w.write(ds.isel(time=slice(1, 2)), 1)
    assert not list(tmp_path.iterdir())


def test_existing_output_is_skipped(tmp_path):
    """An existing output is complete by construction: nothing is pending and the file is left alone."""
    store, stage = NetCDFStore(), _Stage(tmp_path)
    _write(store, stage, _frames(), [N])
    (f_out,) = store.outputs(stage, None)
    mtime = f_out.stat().st_mtime_ns
    with store.writer(stage, None, INPUT) as w:
        assert w.pending_frames(N) == []
    assert f_out.stat().st_mtime_ns == mtime


def test_unknown_option_raises():
    with pytest.raises(pydantic.ValidationError, match="compresion"):
        NetCDFStore(compresion=False)
