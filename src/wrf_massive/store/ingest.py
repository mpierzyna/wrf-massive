# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`IngestStage`: copy existing netCDF post-processing output into a zarr store."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import pathlib

import pydantic

from wrf_massive.base import Simulation, Stage, TPath
from wrf_massive.log import get_logger
from wrf_massive.store.frames import FrameTemplate, frame_times
from wrf_massive.store.netcdf import NetCDFStore
from wrf_massive.store.zarr_store.store import ZarrStore

logger = get_logger("store.ingest")


class IngestStage(Stage):
    """Copy the netCDF output of a postproc stage (`src`) into a zarr store (`store`), one file per worker.

    Files are written in whole files (one shard each), loading one variable at a time, so memory stays at about one
    shard of one variable. Resumes per frame like post-processing into the store. As for post-processing, lay out the
    domain first with `wrf-massive store init-domain --stage <this stage> <sim dirs>`; the template is the first
    netCDF file.
    """

    src: NetCDFStore  # where the netCDF files are (`file_suffix`, optionally `root`: another workspace's sims)
    src_dir: TPath = pathlib.Path("4_postproc")  # work dir of the postproc stage that wrote them, relative to a sim
    store: ZarrStore  # destination (named like `PostProcStage.store`: the store the stage writes to)
    run_parallel: bool = False
    n_workers: int | None = pydantic.Field(default=None, gt=0)  # default: all allocated CPUs

    @pydantic.model_validator(mode="after")
    def ensure_resources_for_parallel(self):
        if self.run_parallel and self.resources is None:
            raise ValueError("'run_parallel=True' requires 'resources' to be set.")
        if self.n_workers is not None and not self.run_parallel:
            raise ValueError("'n_workers' is set but 'run_parallel=False', so it would be ignored.")
        return self

    def get_inputs(self, s: Simulation) -> list[pathlib.Path]:
        inputs = self.src.outputs(self, s, self.src_dir)
        if not inputs:
            raise FileNotFoundError(f"No netCDF output in {self.src.output_dir(self, s, self.src_dir)}.")
        return inputs

    def stored_vars(self, s: Simulation | None = None) -> list[str]:
        """Variables of the netCDF output of `s` (of its first file)."""
        if s is None:
            raise ValueError("The variables of an IngestStage depend on the sim's files.")
        return list(NetCDFStore.read_frames(self.get_inputs(s)[0], 0, 1).data_vars)

    def store_template(self, s: Simulation) -> FrameTemplate | None:
        try:
            inputs = self.get_inputs(s)
        except FileNotFoundError:
            return None
        times = [frame_times(f) for f in inputs]
        steps = [t[1] - t[0] for t in times if len(t) > 1]
        if not steps:
            raise ValueError(f"No netCDF file of {s.name} has two frames to take the output interval from.")
        return FrameTemplate(
            ds=NetCDFStore.read_frames(inputs[0], 0, 1).load(),
            frames_per_file=max(len(t) for t in times),
            interval=steps[0],
            source=inputs[0],
        )

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
            if not self.run_parallel:
                for f in inputs:
                    self.run_single(s, f)
                return
            n_workers = self.n_workers or self.resources.cpus_total
            logger.info(f"Ingesting {len(inputs)} file(s) with {n_workers} workers...")
            ctx = multiprocessing.get_context("spawn")  # see PostProcStage: no fork of a multi-threaded parent
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                futures = [ex.submit(self.run_single, s, f) for f in inputs]
                failures = []
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        logger.error(str(e))
                        failures.append(e)
            if failures:
                raise RuntimeError(
                    f"Ingest failed for {len(failures)} of {len(inputs)} file(s). First error: {failures[0]}"
                ) from failures[0]

    def run_single(self, s: Simulation, f: pathlib.Path):
        try:
            n = len(frame_times(f))
            with self.store.writer(self, s, f) as w:
                pending = w.pending_frames(n)
                if not pending:
                    return
                logger.info(f"Ingesting {f.name} ({len(pending)} of {n} frame(s))...")
                for i0, n_run in _runs(pending):
                    w.write(NetCDFStore.read_frames(f, i0, n_run), i0)  # lazy: loaded per variable by the writer
        except Exception as e:
            raise RuntimeError(f"Ingest failed for {f.name}: {e}") from e


def _runs(frames: list[int]) -> list[tuple[int, int]]:
    """Consecutive runs `(start, length)` of sorted frame indices, e.g. [0, 1, 2, 5] -> [(0, 3), (5, 1)]."""
    runs: list[tuple[int, int]] = []
    for i in frames:
        if runs and runs[-1][0] + runs[-1][1] == i:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((i, 1))
    return runs
