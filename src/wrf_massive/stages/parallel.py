# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stages that process the files of a simulation independently, optionally in a pool of worker processes."""

from __future__ import annotations

import concurrent.futures
import functools
import multiprocessing
import pathlib
from typing import Self

import pydantic

from wrf_massive.base import Simulation, Stage
from wrf_massive.log import get_logger

logger = get_logger("stages")


class FileParallelStage(Stage):
    """A stage whose `process_file(s, f)` handles one input file, run serially or by `n_workers` processes.

    Workers are started with `spawn`, not `fork`: the parent may run threads by then (e.g. zarr's IO threads), and
    forking a multi-threaded process can deadlock the children. So the stage (its class, functions it holds) must be
    importable by module name, and a script that runs it in parallel needs an `if __name__ == "__main__":` guard.
    """

    run_parallel: bool = False  # whether to run in parallel using multiprocessing
    # Number of parallel workers. None uses all allocated CPUs (`resources.cpus_total`). Set it lower to give each
    # worker more memory, e.g. on clusters with a minimum allocation or a fixed memory per CPU.
    n_workers: int | None = pydantic.Field(default=None, gt=0)

    @pydantic.model_validator(mode="after")
    def ensure_resources_for_parallel(self) -> Self:
        """A parallel run takes its worker count from `resources`, so refuse to build a stage that cannot."""
        if self.run_parallel and self.resources is None:
            raise ValueError(
                "'run_parallel=True' requires 'resources' to be set: "
                "the number of workers is taken from 'resources.cpus_total'."
            )
        if self.n_workers is not None:
            if not self.run_parallel:
                raise ValueError("'n_workers' is set but 'run_parallel=False', so it would be ignored.")
            if self.n_workers > self.resources.cpus_total:
                raise ValueError(
                    f"'n_workers={self.n_workers}' exceeds the allocated CPUs "
                    f"(resources.cpus_total={self.resources.cpus_total})."
                )
        return self

    def process_file(self, s: Simulation, f: pathlib.Path) -> None:
        raise NotImplementedError

    def run_single(self, s: Simulation, f: pathlib.Path) -> None:
        """Process one file; a failure names the file, which is otherwise lost when it surfaces from a worker."""
        try:
            self.process_file(s, f)
        except Exception as e:
            raise RuntimeError(f"{type(self).__name__} failed for {f.name}: {e}") from e

    def process_files(self, s: Simulation, files: list[pathlib.Path]) -> None:
        """`run_single` for every file, serially or in the worker pool. All failures are collected and reported."""
        if not self.run_parallel:
            logger.info(f"Processing {len(files)} file(s) serially...")
            for f in files:
                self.run_single(s, f)
            return

        n_workers = self.n_workers or self.resources.cpus_total  # resources ensured by the validator
        logger.info(f"Processing {len(files)} file(s) with {n_workers} workers...")
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = [ex.submit(functools.partial(self.run_single, s), f) for f in files]
            # Collect all failures instead of raising the first one: the pool waits for every queued future on exit
            # anyway, so aborting early saves no time but loses the other error messages.
            failures = []
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logger.error(str(e))
                    failures.append(e)
        if failures:
            raise RuntimeError(
                f"{type(self).__name__} failed for {len(failures)} of {len(files)} file(s). First error: {failures[0]}"
            ) from failures[0]
        logger.info("Done!")
