from __future__ import annotations

import concurrent.futures
import copy
import datetime
import os
import pathlib
import re
import shutil
from typing import Callable, Dict, List, Self, Tuple

import netCDF4
import pydantic
import wrf
import xarray as xr

from wrf_massive.base import Simulation, Stage, TPath
from wrf_massive.log import get_logger

logger = get_logger("stages.postproc")

STAGE_DIR = pathlib.Path(os.path.dirname(__file__))

# Type alias for list of variable names to extract from wrfout files.
# If tuple, first element is variable name, second element is dimension along which to destagger.
TVarList = List[str | Tuple[str, str]]


def load_wrfout(f: pathlib.Path | str, vars_to_extract: TVarList) -> xr.Dataset:
    """Load wrfout file and extract variables using wrf-python."""
    wrf_file = netCDF4.Dataset(f)

    # Start actual extraction
    res = {}
    for v in vars_to_extract:
        # Optionally: Extract dimension along which data will be destaggered
        if isinstance(v, tuple):
            v, destagger_dim = v
        else:
            destagger_dim = None

        # Get variable for all timesteps (ATTENTION! This is heavy on memory!)
        logger.info(f"Reading {v}... ")
        v_data = wrf.getvar(wrf_file, v, timeidx=wrf.ALL_TIMES)

        # Now destagger
        if destagger_dim:
            destagger_dim_ind = v_data.dims.index(destagger_dim)
            logger.info(f"Destaggering along {destagger_dim} ({destagger_dim_ind})... ")
            v_data = wrf.destagger(v_data, destagger_dim_ind, meta=True)

        # Serialise object attributes for later netcdf storage
        v_data.attrs["projection"] = v_data.attrs["projection"].proj4()
        v_data = v_data.drop_vars("latlon_coord", errors="ignore")

        # Store it. Maybe write to output file directly to release RAM
        assert v not in res, f"Variable {v} already in results!"  # this shouldn't happen
        res[v] = v_data
        logger.info("Done!")

    # Collect everything into a single dataset
    res_ds = xr.Dataset(res)
    res_ds = res_ds.drop_vars("latlon_coord", errors="ignore")

    # Copy attributes from original file
    for attr in wrf_file.ncattrs():
        res_ds.attrs[attr] = wrf_file.getncattr(attr)

    # Rename time
    res_ds = res_ds.rename({"Time": "time"})

    return res_ds


class PostProcFn(pydantic.BaseModel):
    """Post-processing function with metadata about required variables and output variable names."""

    fn: Callable[[xr.Dataset], Dict[str, xr.DataArray]]
    returns: List[str]
    requires: TVarList = []

    def __call__(self, ds: xr.Dataset) -> Dict[str, xr.DataArray]:
        """Take the current state of postproc dataset and return a dict of new variables to add to the dataset."""
        return self.fn(ds)


def _w_rename_fn(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """Rename wa to w."""
    return {"w": ds["wa"]}


fn_w_rename = PostProcFn(
    requires=["wa"],
    returns=["w"],
    fn=_w_rename_fn,
)


def _split_uvmet(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """Split uvmet into u_met and v_met."""
    return {"u_met": ds["uvmet"][0], "v_met": ds["uvmet"][1]}


fn_uv_split = PostProcFn(
    requires=["uvmet"],
    returns=["u_met", "v_met"],
    fn=_split_uvmet,
)


class PostProcStage(Stage):
    """Post-processing stage for wrfout files. Extracts variables and applies post-processing functions."""

    wrfout_dir: TPath  # directory with wrfout files (relative to sim_dir)
    domain: int  # domain to process, e.g. 1 for d01 (1-indexed!)

    # Variables to extract from wrfout files using wrf-python.
    # ALL-CAPS vars taken from wrfout directly, lower-case vars are processed by wrf.getvar()
    extract_vars: TVarList

    # List of post-processing functions and their required variables
    # Each entry is a tuple of (function_name, function_callable, optional list_of_required_vars).
    # Postproc function takes postproc dataset and returns dict of new variables to add to dataset.
    postproc_fns: List[PostProcFn] = []

    compression: bool
    run_parallel: bool = False  # whether to run in parallel using multiprocessing
    file_suffix: str = "proc"  # appended to wrfout filenames to indicate post-processed files
    discard_warmup: bool = True  # whether to discard wrfout files from warmup period (default: True)

    @pydantic.field_validator("extract_vars", mode="after")
    def ensure_defaults(cls, extract_vars: TVarList) -> TVarList:
        """Ensure reasonable defaults like pressure level height and terrain height are included."""
        defaults = ["z", "HGT", "p"]
        for v in defaults:
            if v not in extract_vars:
                extract_vars.append(v)
                logger.warning(f"Added variable '{v}' as reasonable aux variable.")

        # Make sure we have full wind vector if one component is requested.
        if "u_met" in extract_vars and "v_met" not in extract_vars:
            extract_vars.append("v_met")
            logger.warning("Adding 'v_met' because 'u_met' is requested.")
        if "v_met" in extract_vars and "u_met" not in extract_vars:
            extract_vars.append("u_met")
            logger.warning("Adding 'u_met' because 'v_met' is requested.")

        return extract_vars

    @pydantic.model_validator(mode="after")
    @classmethod
    def set_default_postproc_fns(cls, obj: Self) -> Self:
        fns = []

        # in wrfout, w=wa, but we want it as w
        if "w" in obj.extract_vars:
            fns.append(fn_w_rename)
            obj.extract_vars.remove("w")  # will be added by postproc function, so remove from extract_vars

        # wrf.getvar() returns u_met and v_met in one variable, so we need to add a postproc function to split them
        if "u_met" in obj.extract_vars or "v_met" in obj.extract_vars:
            fns.append(fn_uv_split)
            if "u_met" in obj.extract_vars:
                obj.extract_vars.remove("u_met")
            if "v_met" in obj.extract_vars:
                obj.extract_vars.remove("v_met")

        # Add default postproc fns to the top, so downstream fns can depend on them
        obj.postproc_fns = [*fns, *obj.postproc_fns]
        return obj

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
            inputs_filtered = []
            inputs_begin = [_parse_fname(f.name) for f in inputs]
            inputs_begin = list(zip(inputs, inputs_begin))  # (fname, begin)

            # Simple check if only single file
            if len(inputs) == 1:
                fname, fdate = inputs_begin[0]
                if fdate < s.begin:
                    logger.warning(
                        f"Only single file {fname} found, which seems to include warmup (expected start: {s.begin}). "
                        f"File is kept, but warmup is likely not discarded!"
                    )
                    inputs_filtered.append(fname)
                elif fdate == s.begin:
                    inputs_filtered.append(fname)
                else:
                    logger.warning(f"Only single file {fname} found, but starts after simulation begin {s.begin}.")
                    inputs_filtered.append(fname)

                return inputs_filtered

            # For multiple files, sophisticated check that keeps files around and after simulation begin
            for (a_path, a_begin), (b_path, b_begin) in zip(inputs_begin[:-1], inputs_begin[1:]):
                if a_begin < s.begin < b_begin:
                    # If begin falls between a and b, also keep a.
                    inputs_filtered.append(a_path)
                    inputs_filtered.append(b_path)
                else:
                    # Either both before begin or both after begin. Keep only if AT begin or after
                    if a_begin >= s.begin:
                        inputs_filtered.append(a_path)
                    if b_begin >= s.begin:
                        inputs_filtered.append(b_path)

            return inputs_filtered

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
        if self.run_parallel:
            # Parallel run
            n_workers = self.resources.cpus_total
            logger.info(f"Starting parallel post-processing with {n_workers} workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(self.run_single, s, f) for f in self.get_inputs(s)]
                for f in concurrent.futures.as_completed(futures):
                    f.result()  # Raise exceptions if any
            logger.info("Done!")
        else:
            # Serial run
            logger.info("Starting serial post-processing...")
            for f in self.get_inputs(s):
                self.run_single(s, f)

    def run_single(self, s: Simulation, i_f: int | pathlib.Path):
        f = self.get_inputs(s)[i_f] if isinstance(i_f, int) else i_f
        f_out = (
            self.get_work_dir(s) / f"{f.name.replace(':', '-')}_{self.file_suffix}.nc"
        )  # replace `:` to avoid problems with windows storage

        if f_out.exists():
            logger.info(f"Output file {f_out} already exists. Skipping.")
            return

        # Collect dependency variables from postproc_fns
        extract_vars = copy.deepcopy(self.extract_vars)
        for fn in self.postproc_fns:
            # Add required variables to extract_vars if not already present
            for v in fn.requires:
                if v not in extract_vars:
                    extract_vars.append(v)

        # Load wrfout file
        logger.info(f"Post-processing {f.name}...")
        ds = load_wrfout(f, vars_to_extract=extract_vars)

        # Apply postproc functions
        for fn in self.postproc_fns:
            logger.info(f"Applying post-processing function for {fn.returns}...")
            new_vars = fn(ds)
            for name, data in new_vars.items():
                ds[name] = data

        # Select only vars requested explicitly OR from postproc
        extract_vars = [v[0] if isinstance(v, tuple) else v for v in self.extract_vars]  # remove destagger dim
        for fn in self.postproc_fns:
            extract_vars.extend(fn.returns)
        ds = ds[extract_vars]

        # Log diagnostics
        logger.debug(ds.sizes)
        logger.debug(ds.dtypes)
        logger.info(f"-> Processing done. Expected filesize: {ds.nbytes / 1e9:.1f} GB")

        # Save
        logger.info(f"-> Saving to {f_out}{' (compressed)' if self.compression else ''}...")
        encoding = {}
        if self.compression:
            encoding = {var: {"zlib": True} for var in ds.data_vars}
        ds.to_netcdf(f_out, engine="h5netcdf", encoding=encoding)

    def is_done(self, s: Simulation) -> bool:
        try:
            n_expected = len(self.get_inputs(s))
        except FileNotFoundError:
            # No inputs found -> not done
            return False
        n_done = len(list(self.get_work_dir(s).glob(f"wrfout_*{self.file_suffix}.nc")))
        return n_done == n_expected
