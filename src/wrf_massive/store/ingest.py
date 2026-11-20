# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`IngestStage`: copy existing netCDF post-processing output into a zarr store."""

from __future__ import annotations

import pathlib

import numpy as np
import pydantic
import xarray as xr

from wrf_massive.base import Simulation, TPath
from wrf_massive.log import get_logger
from wrf_massive.stages.parallel import FileParallelStage
from wrf_massive.store.frames import FrameTemplate, crop_xy, frame_times, read_static
from wrf_massive.store.netcdf import NetCDFStore
from wrf_massive.store.zarr_store.store import ZarrStore

logger = get_logger("store.ingest")


class IngestStage(FileParallelStage):
    """Copy the netCDF output of a postproc stage (`src`) into a zarr store (`store`), one file per worker.

    Files are written in whole files (one shard each), loading one variable at a time, so memory stays at about one
    shard of one variable. Resumes per frame like post-processing into the store. As for post-processing, lay out the
    domain first with `wrf-massive store init-domain --stage <this stage> <sim dirs>`; the template is the first
    netCDF file.
    """

    src: NetCDFStore  # where the netCDF files are (`file_suffix`, optionally `root`: another workspace's sims)
    src_dir: TPath = pathlib.Path("4_postproc")  # work dir of the postproc stage that wrote them, relative to a sim
    store: ZarrStore  # destination (named like `PostProcStage.store`: the store the stage writes to)
    # Cells cropped from each side in y and x (`[crop:-crop]`), as `PostProcStage.crop`: to drop WRF's lateral
    # boundary zone from output that was post-processed without it. 0: as in the netCDF files.
    crop: int = pydantic.Field(default=0, ge=0)

    def get_inputs(self, s: Simulation) -> list[pathlib.Path]:
        inputs = self.src.outputs(self, s, self.src_dir)
        if not inputs:
            raise FileNotFoundError(f"No netCDF output in {self.src.output_dir(self, s, self.src_dir)}.")
        return inputs

    def stored_vars(self, s: Simulation) -> list[str]:
        """Variables of the netCDF output of `s` (of its first file)."""
        return list(NetCDFStore.read_frames(self.get_inputs(s)[0], 0, 1).data_vars)

    def store_template(self, s: Simulation) -> FrameTemplate | None:
        try:
            inputs = self.get_inputs(s)
        except FileNotFoundError:
            return None
        return FrameTemplate.from_inputs(inputs, lambda f: self._read(f, 0, 1).load())

    def input_static(self, f: pathlib.Path, names: list[str]) -> dict[str, np.ndarray]:
        """Static (y, x) fields of netCDF file `f` as ingested (cropped); names not in `f` are skipped."""
        return read_static(f, names, self.crop)

    def _read(self, f: pathlib.Path, i0: int, n: int | None) -> xr.Dataset:
        return crop_xy(NetCDFStore.read_frames(f, i0, n), self.crop)  # lazy: cropping selects, nothing is loaded

    def is_setup(self, s: Simulation) -> bool:
        return self.get_work_dir(s, create=False).is_dir()

    def setup(self, s: Simulation):
        self.get_work_dir(s)  # created; holds the sim lock

    def is_done(self, s: Simulation) -> bool:
        try:
            inputs = self.get_inputs(s)
        except FileNotFoundError:
            return False
        return self.store.is_done(self, s, inputs)

    def run(self, s: Simulation):
        inputs = self.get_inputs(s)
        with self.store.session(self, s, inputs):
            self.process_files(s, inputs)

    def process_file(self, s: Simulation, f: pathlib.Path):
        n = len(frame_times(f))
        with self.store.writer(self, s, f) as w:
            pending = w.pending_frames(n)
            if not pending:
                return
            logger.info(f"Ingesting {f.name} ({len(pending)} of {n} frame(s))...")
            for i0, n_run in _runs(pending):
                w.write(self._read(f, i0, n_run), i0)  # lazy: loaded per variable by the writer


def _runs(frames: list[int]) -> list[tuple[int, int]]:
    """Consecutive runs `(start, length)` of sorted frame indices, e.g. [0, 1, 2, 5] -> [(0, 3), (5, 1)]."""
    runs: list[tuple[int, int]] = []
    for i in frames:
        if runs and runs[-1][0] + runs[-1][1] == i:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((i, 1))
    return runs
