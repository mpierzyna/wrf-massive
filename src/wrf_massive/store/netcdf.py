# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""netCDF output: one file per input file, in the stage's work dir."""

from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import h5netcdf
import numpy as np
import xarray as xr

from wrf_massive.base import TPath
from wrf_massive.log import get_logger
from wrf_massive.store.base import PostProcStore, Writer

if TYPE_CHECKING:
    from wrf_massive.base import Simulation, Stage

logger = get_logger("store.netcdf")


class NetCDFStore(PostProcStore):
    """One netCDF file per input file, named `<input name>_<file_suffix>.nc`, in the stage's work dir.

    A file is written frame by frame into a hidden temp file and renamed when complete, so an existing output file is
    always complete: it is skipped on a re-run and counts towards `is_done`. Its files can also be read frame by frame
    (`read_frames`), e.g. by `IngestStage`.
    """

    kind: Literal["netcdf"] = "netcdf"
    compression: bool = True  # zlib
    file_suffix: str = "proc"  # appended to the input filename to mark post-processed files
    # If set, files are in `<root>/<sim name>/<work dir>` instead of the sim's own work dir, e.g. to read the output
    # of the same sims in another (old) workspace
    root: TPath | None = None

    def output_dir(self, stage: Stage, s: Simulation, work_dir: pathlib.Path | None = None) -> pathlib.Path:
        """Dir of the files: the stage's work dir (or `work_dir`, relative to the sim dir), below `root` if set."""
        if self.root is not None:
            return pathlib.Path(self.root) / s.name / (work_dir or stage.work_dir)
        return stage.get_work_dir(s) if work_dir is None else s.sim_dir / work_dir

    def output_path(self, stage: Stage, s: Simulation, f: pathlib.Path) -> pathlib.Path:
        # Replace `:` of the WRF date in the input name, which causes problems on Windows storage
        return self.output_dir(stage, s) / f"{f.name.replace(':', '-')}_{self.file_suffix}.nc"

    def outputs(self, stage: Stage, s: Simulation, work_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
        """Complete output files of simulation `s` (temp files start with a dot and are not matched)."""
        return sorted(self.output_dir(stage, s, work_dir).glob(f"wrfout_*{self.file_suffix}.nc"))

    @staticmethod
    def read_frames(f: pathlib.Path, i0: int = 0, n: int | None = None) -> xr.Dataset:
        """Frames `i0 .. i0 + n` of output file `f`, lazily loaded (one variable at a time when accessed).

        Also reads output of older wrf-massive versions, whose single-frame files have a scalar `time`.
        """
        ds = xr.open_dataset(f, engine="h5netcdf")
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")  # to every data variable; coordinates (XLAT/XLONG) stay as they are
        return ds.isel(time=slice(i0, None if n is None else i0 + n))

    def is_done(self, stage: Stage, s: Simulation, inputs: list[pathlib.Path]) -> bool:
        return all(self.output_path(stage, s, f).exists() for f in inputs)

    @contextlib.contextmanager
    def writer(self, stage: Stage, s: Simulation, f: pathlib.Path) -> Iterator[Writer]:
        f_out = self.output_path(stage, s, f)
        f_out.parent.mkdir(parents=True, exist_ok=True)  # below `root`, not created as a work dir
        w = _NetCDFWriter(f_out, compression=self.compression)
        try:
            yield w
            w.commit()
        finally:
            w.f_tmp.unlink(missing_ok=True)


class _NetCDFWriter(Writer):
    def __init__(self, f_out: pathlib.Path, compression: bool):
        self.f_out = f_out
        self.f_tmp = f_out.with_name(f".{f_out.name}.tmp")  # leading dot keeps it out of the output globs
        self.compression = compression
        self._n: int | None = None  # number of frames to write, set by `pending_frames`
        self._written = 0  # frames written to the temp file so far

    def pending_frames(self, n: int) -> list[int]:
        # A file is complete or absent (temp file + rename), so it never resumes mid-file
        if self.f_out.exists():
            logger.info(f"Output file {self.f_out} already exists. Skipping.")
            self._n = 0
            return []
        self._n = n
        return list(range(n))

    def write(self, ds: xr.Dataset, i0: int) -> None:
        if i0 != self._written:
            raise ValueError(f"Frames must be written in order: expected frame {self._written}, got {i0}.")
        if i0 == 0:
            n_total = self._n if self._n is not None else ds.sizes["time"]
            logger.info(f"-> Saving to {self.f_out}{' (compressed)' if self.compression else ''}...")
            logger.info(f"-> Expected filesize: {n_total * ds.nbytes / ds.sizes['time'] / 1e9:.1f} GB")
            ds.to_netcdf(self.f_tmp, engine="h5netcdf", encoding=self._encoding(ds), unlimited_dims=["time"])
        else:
            _append_frames(self.f_tmp, ds, i0)
        self._written += ds.sizes["time"]

    def commit(self) -> None:
        """Rename the complete temp file to the output file (atomic within the same directory)."""
        if self._n == 0:  # output existed already
            return
        if self._n is not None and self._written != self._n:
            raise RuntimeError(f"{self.f_out.name}: only {self._written} of {self._n} frames written.")
        self.f_tmp.replace(self.f_out)

    def _encoding(self, ds: xr.Dataset) -> dict:
        """netCDF encoding for the first frame(s), which fixes it for the frames appended to the file later."""
        # Explicit time units with integer seconds, so the appended frames can be encoded the same way.
        t0 = np.datetime_as_string(ds["time"].values[0], unit="s").replace("T", " ")
        encoding = {"time": {"units": f"seconds since {t0}", "dtype": "int64"}}
        for v in ds.data_vars:
            enc = {}
            if "time" in ds[v].dims:  # one chunk per frame: appending touches only new chunks
                enc["chunksizes"] = tuple(1 if d == "time" else ds.sizes[d] for d in ds[v].dims)
            if self.compression:
                enc["zlib"] = True
            encoding[v] = enc
        return encoding


def _append_frames(f: pathlib.Path, ds: xr.Dataset, i0: int):
    """Write the frames of `ds` as frames `i0, i0 + 1, ...` into `f`, which `to_netcdf` created for frame 0."""
    n = ds.sizes["time"]
    with h5netcdf.File(f, "a") as nc:
        nc.resize_dimension("time", i0 + n)
        time = nc.variables["time"]
        t0 = np.datetime64(time.attrs["units"].removeprefix("seconds since "), "s")
        time[i0 : i0 + n] = (ds["time"].values - t0) // np.timedelta64(1, "s")
        for v in ds.data_vars:
            if "time" not in ds[v].dims:
                continue  # time-invariant: written with frame 0
            var = nc.variables[v]
            assert var.dimensions == ds[v].dims and ds[v].dims[0] == "time", f"{v}: {var.dimensions}, {ds[v].dims}"
            var[i0 : i0 + n, ...] = ds[v].values
