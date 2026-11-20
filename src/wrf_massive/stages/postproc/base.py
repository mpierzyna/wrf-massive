# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Post-processing of wrfout files. See `variables.py` for how variables are named."""

from __future__ import annotations

import concurrent.futures
import datetime
import os
import pathlib
import re
import shutil
from typing import Callable, Dict, List, Self

import numpy as np
import pydantic
import xarray as xr

from wrf_massive.base import Simulation, Stage, TPath, TPathExists
from wrf_massive.log import get_logger
from wrf_massive.stages.postproc.iofields import parse_myoutfields
from wrf_massive.stages.postproc.variables import REQUIRED_WRF_FIELDS, TVarList, var_name, wrf_fields_of
from wrf_massive.stages.postproc.wrfout import load_wrfout, n_frames
from wrf_massive.store.netcdf import NetCDFStore

logger = get_logger("stages.postproc")

STAGE_DIR = pathlib.Path(os.path.dirname(__file__))


class PostProcFn(pydantic.BaseModel):
    """Post-processing function with metadata about required variables and output variable names.

    Files are processed one time step (frame) at a time, so `fn` sees a dataset with a `time` dimension of length 1
    and must not need other time steps (e.g. no time derivatives or averages).
    """

    fn: Callable[[xr.Dataset], Dict[str, xr.DataArray]]
    returns: List[str]
    requires: TVarList = []

    def __call__(self, ds: xr.Dataset) -> Dict[str, xr.DataArray]:
        """Take the current state of postproc dataset and return a dict of new variables to add to the dataset."""
        return self.fn(ds)


class PostProcStage(Stage):
    """Post-processing stage for wrfout files. Extracts variables and applies post-processing functions.

    Every variable needed (in `extract_vars` or required by a postproc fn) that no postproc fn returns is read from the
    wrfout file: WRF fields as they are, wrf-python derived variables via `wrf.getvar()` (see `variables.py`). Then the
    postproc fns run in order, each seeing what the file provided and what the fns before it returned.

    Everything about writing the result (format, location, encoding, resuming) is the job of `store`.
    """

    wrfout_dir: TPath  # directory with wrfout files (relative to sim_dir)
    domain: int  # domain to process, e.g. 1 for d01 (1-indexed!)

    # Variables to extract from wrfout files and store in the output: WRF fields (ALL-CAPS) or derived variables.
    extract_vars: TVarList

    # Post-processing functions, applied in order. Postproc fn takes postproc dataset and returns dict of new
    # variables to add to dataset. wrf-python derived variables they require are read from the file beforehand.
    postproc_fns: List[PostProcFn] = []

    # iofields file of the WRF run (usually `myoutfields.txt`, as in `WRFStage.myoutfields_path`). If given, it is
    # checked at init that WRF writes every WRF field needed for post-processing to `iofields_stream`.
    myoutfields_path: TPathExists | None = None
    iofields_stream: int = 7  # history stream of the post-processed `wrfout_aux` files (auxhist7)

    # Where the output goes; the stage only reads and processes
    store: NetCDFStore = pydantic.Field(default_factory=NetCDFStore)

    run_parallel: bool = False  # whether to run in parallel using multiprocessing
    # Number of parallel workers. None uses all allocated CPUs (`resources.cpus_total`). Set it lower to give each
    # worker more memory, e.g. on clusters with a minimum allocation or a fixed memory per CPU.
    n_workers: int | None = pydantic.Field(default=None, gt=0)
    discard_warmup: bool = True  # whether to discard wrfout files from warmup period (default: True)

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

    # NOTE: The validators below must be model validators (not field validators), because field validators are
    # skipped for default values. Subclasses like Cn2PostProcStage define `extract_vars` as a class default, so a
    # field validator would never see it. Model validators run in definition order, so `ensure_defaults` fills in
    # the defaults before `check_vars` checks them.
    @pydantic.model_validator(mode="after")
    def ensure_defaults(self) -> Self:
        """Ensure reasonable defaults like pressure level height and terrain height are included."""
        defaults = ["z", "HGT", "p"]
        for v in defaults:
            if v not in self.extract_vars:
                self.extract_vars.append(v)
                logger.warning(f"Added variable '{v}' as reasonable aux variable.")

        # Make sure we have full wind vector if one component is requested.
        if "u_met" in self.extract_vars and "v_met" not in self.extract_vars:
            self.extract_vars.append("v_met")
            logger.warning("Adding 'v_met' because 'u_met' is requested.")
        if "v_met" in self.extract_vars and "u_met" not in self.extract_vars:
            self.extract_vars.append("u_met")
            logger.warning("Adding 'u_met' because 'v_met' is requested.")

        return self

    @pydantic.model_validator(mode="after")
    def check_vars(self) -> Self:
        """Check at init that every variable needed can be read from the wrfout files or is returned in time."""
        self._check_wrf_fields_written()
        self._check_postproc_fn_order()
        return self

    def vars_to_read(self) -> TVarList:
        """Variables read from the wrfout file: everything needed that no postproc fn returns (unique, in order)."""
        returned = {name for fn in self.postproc_fns for name in fn.returns}
        needed = [*self.extract_vars, *(v for fn in self.postproc_fns for v in fn.requires)]
        to_read = {}
        for v in needed:
            name = var_name(v)
            if name in returned:
                continue
            if to_read.setdefault(name, v) != v:  # e.g. EL_PBL once as it is, once destaggered
                raise ValueError(
                    f"'{name}' is needed both as {to_read[name]!r} and as {v!r}; it can be read only one way."
                )
        return list(to_read.values())

    def _check_wrf_fields_written(self):
        """First check: WRF writes every WRF field needed to read `vars_to_read` (per `myoutfields_path`)."""
        needed_for = {f: "coordinates" for f in REQUIRED_WRF_FIELDS}  # WRF field -> variable it is needed for
        for name in map(var_name, self.vars_to_read()):
            for f in wrf_fields_of(name):  # raises for unknown derived variables, also without `myoutfields_path`
                needed_for.setdefault(f, name)

        if self.myoutfields_path is None:
            logger.warning(
                f"No 'myoutfields_path' set for {self.name}: cannot check that WRF writes {list(needed_for)}."
            )
            return

        written = parse_myoutfields(self.myoutfields_path, self.iofields_stream)
        missing = [f if v == f else f"{f} (for {v})" for f, v in needed_for.items() if f.upper() not in written]
        if missing:
            raise ValueError(
                f"WRF fields needed for post-processing are not written to history stream {self.iofields_stream} "
                f"according to '{self.myoutfields_path}': {', '.join(missing)}. Add them there or drop the "
                f"variables and postproc fns that need them."
            )

    def _check_postproc_fn_order(self):
        """Second check: step through the postproc fns, each may only require what is available before it."""
        available = {var_name(v) for v in self.vars_to_read()}
        for fn in self.postproc_fns:
            missing = [var_name(v) for v in fn.requires if var_name(v) not in available]
            if missing:
                raise ValueError(
                    f"Postproc fn '{fn.fn.__name__}' requires {missing}, returned only by a later postproc fn. "
                    f"Reorder `postproc_fns`."
                )
            available |= set(fn.returns)

    def get_inputs(self, s: Simulation) -> List[pathlib.Path]:
        """Get wrfout files for post-processing. Discard files from warmup period by default."""
        fname_base = f"wrfout_aux_d{self.domain:02d}_"
        re_date = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}")

        def _parse_fname(fname: str) -> datetime.datetime:
            fname_date = re_date.search(fname).group(0)
            fname_date = datetime.datetime.strptime(fname_date, "%Y-%m-%d_%H:%M:%S")
            return fname_date

        logger.debug(f"Using pattern '{fname_base}*' to find dependencies.")
        inputs = list(sorted((s.sim_dir / self.wrfout_dir).glob(f"{fname_base}*")))

        if len(inputs) == 0:
            raise FileNotFoundError(
                f"No wrfout files found in {s.sim_dir / self.wrfout_dir} with pattern '{fname_base}*'."
            )

        if self.discard_warmup:
            inputs_begin = [(f, _parse_fname(f.name)) for f in inputs]  # (fname, begin)

            at_or_after = [f for f, fdate in inputs_begin if fdate >= s.begin]
            before = [f for f, fdate in inputs_begin if fdate < s.begin]

            if not at_or_after:
                # Everything starts before begin -> last file is the one containing the usable period
                logger.warning(
                    f"All {len(inputs)} file(s) start before simulation begin {s.begin}. Keeping last one "
                    f"({before[-1].name}), but warmup is likely not discarded!"
                )
                return before[-1:]

            # If no file starts exactly at begin, the last file before it contains the simulation start
            if before and not any(fdate == s.begin for _, fdate in inputs_begin):
                return [before[-1], *at_or_after]

            return at_or_after

        return inputs

    def setup(self, s: Simulation):
        logger.info("Setting up Cn2 post-processing working dir...")
        work_dir = self.get_work_dir(s)
        shutil.copy(STAGE_DIR / "gitignore", work_dir / ".gitignore")
        logger.info(f"-> .gitignore copied to {work_dir}")
        logger.info("-> Setup done.")

    def is_setup(self, s: Simulation) -> bool:
        return all([(self.get_work_dir(s) / ".gitignore").exists()])

    def run(self, s: Simulation):
        # Deduplicate defensively: two workers writing the same output file corrupt it / fail on file locking
        inputs = list(dict.fromkeys(self.get_inputs(s)))
        self.store.preflight(self, s, inputs)  # fail before any processing if the output cannot be written

        if self.run_parallel:
            # Parallel run
            # `resources` guaranteed to exist by `ensure_resources_for_parallel`
            n_workers = self.n_workers or self.resources.cpus_total
            logger.info(f"Starting parallel post-processing of {len(inputs)} file(s) with {n_workers} workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(self.run_single, s, f) for f in inputs]

                # Collect all failures instead of raising the first one: the pool waits for every queued
                # future on exit anyway, so aborting early saves no time but loses the other error messages.
                failures = []
                for f in concurrent.futures.as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        logger.error(str(e))
                        failures.append(e)

            if failures:
                raise RuntimeError(
                    f"Post-processing failed for {len(failures)} of {len(inputs)} file(s). First error: {failures[0]}"
                ) from failures[0]
            logger.info("Done!")
        else:
            # Serial run
            logger.info(f"Starting serial post-processing of {len(inputs)} file(s)...")
            for f in inputs:
                self.run_single(s, f)

    def run_single(self, s: Simulation, i_f: int | pathlib.Path):
        """Post-process a single wrfout file, given either as path or as index into `get_inputs`."""
        f = self.get_inputs(s)[i_f] if isinstance(i_f, int) else i_f
        try:
            self._run_single(s, f)
        except Exception as e:
            # Attach the input file, which is otherwise lost when the error surfaces in the parent process
            raise RuntimeError(f"Post-processing failed for {f.name}: {e}") from e

    def _run_single(self, s: Simulation, f: pathlib.Path):
        # Process one frame at a time, so memory is bounded by the size of a single frame, not by the number of
        # frames per file. The store's writer decides which frames are still missing and commits on success.
        n = n_frames(f)
        with self.store.writer(self, s, f) as w:
            pending = w.pending_frames(n)
            if not pending:
                return
            logger.info(f"Post-processing {f.name} ({len(pending)} of {n} frame(s))...")
            for i in pending:
                w.write(self._process_frame(f, i), i)

    def _process_frame(self, f: pathlib.Path, timeidx: int | None) -> xr.Dataset:
        """Read frame `timeidx` (None: all frames at once) of wrfout file `f`, apply the postproc fns and keep only
        the stored variables."""
        ds = load_wrfout(f, vars_to_extract=self.vars_to_read(), timeidx=timeidx)

        # Apply postproc functions
        for fn in self.postproc_fns:
            logger.info(f"Applying post-processing function for {fn.returns}...")
            new_vars = fn(ds)
            for name, data in new_vars.items():
                ds[name] = data

        # Select only vars requested explicitly OR returned by the stage's own postproc fns
        keep = [var_name(v) for v in self.extract_vars]
        keep += [name for fn in self.postproc_fns for name in fn.returns]
        ds = ds[list(dict.fromkeys(keep))]

        # Variables read are float32 already (see `read_var`); this catches postproc fn results computed in float64.
        for v in ds.data_vars:
            if ds[v].dtype == np.float64:
                ds[v] = ds[v].astype(np.float32)

        logger.debug(ds.sizes)
        logger.debug(ds.dtypes)
        return ds

    def is_done(self, s: Simulation) -> bool:
        try:
            inputs = self.get_inputs(s)
        except FileNotFoundError:
            # No inputs found -> not done
            return False
        return self.store.is_done(self, s, inputs)
