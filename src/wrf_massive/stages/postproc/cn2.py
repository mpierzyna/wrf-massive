from __future__ import annotations

import concurrent.futures
import copy
import os
import datetime
import pydantic
import pathlib
from typing import List, Tuple
import numpy as np
import shutil

from wrf_massive.base import Simulation, Stage, TPath
from wrf_massive.log import get_logger
from wrf_massive.stages.postproc.utils import load_wrfout

logger = get_logger("stages.postproc.cn2")
STAGE_DIR = pathlib.Path(os.path.dirname(__file__))


def get_ct2_hb15(*, var_theta, Lm):
    """Variance-based CT2 parameterization from He and Basu (2015).

    Parameters
    ----------
    var_theta : float
        Variance of potential temperature, K^2.
    Lm : float
        Master length scale, m.

    Returns
    -------
    CT2 : float
        CT2 estimate, K^2 m^(-2/3).
    """
    # Coefficents from LES
    B1 = 24
    B2 = 15

    # Clip Lm to avoid division by zero
    Lm = np.clip(Lm, a_min=1e-4, a_max=None)

    ct2 = 3.2 * B1 ** (1 / 3) / B2 * Lm ** (-2 / 3) * var_theta
    return ct2


def gladstone_cn2_simple(*, ct2, p_hPa, t_K):
    """Simplified Gladstone equation without humidity correction."""
    cn2 = (7.9e-5 * p_hPa / t_K**2) ** 2 * ct2
    return cn2


class PostprocCn2Stage(Stage):
    wrfout_dir: TPath  # directory with wrfout files (relative to sim_dir)
    domain: int  # domain to process, e.g. 1 for d01 (1-indexed!)
    extract_vars: List[str | Tuple[str, str]]  # variables to extract from wrfout files
    compression: bool
    run_parallel: bool = False  # whether to run in parallel using multiprocessing

    @pydantic.field_validator("extract_vars", mode="after")
    def ensure_defaults(cls, extract_vars: List[str]) -> List[str]:
        """Ensure reasonable defaults like pressure level height and terrain height are included."""
        defaults = ["z", "HGT", "p"]
        for v in defaults:
            if v not in extract_vars:
                extract_vars.append(v)
                logger.warning(f"Added variable '{v}' as reasonable aux variable for Cn2 computation.")
        return extract_vars

    def get_inputs(self, s: Simulation, discard_warmup: bool = True) -> List[pathlib.Path]:
        """Get wrfout files for post-processing. Discard files from warmup period by default."""
        fname_base = f"wrfout_aux_d{self.domain:02d}_"

        def _parse_fname(fname: str) -> datetime.datetime:
            fname_date = fname.replace(fname_base, "")
            fname_date = datetime.datetime.strptime(fname_date, "%Y-%m-%d_%H:%M:%S")
            return fname_date

        logger.debug(f"Using pattern '{fname_base}*' to find dependencies.")
        inputs = list(sorted((s.sim_dir / self.wrfout_dir).glob(f"{fname_base}*")))

        if len(inputs) == 0:
            raise FileNotFoundError(
                f"No wrfout files found in {s.sim_dir / self.wrfout_dir} with pattern '{fname_base}*'."
            )

        if discard_warmup:
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
                    # Either both before begin or both after begin...
                    if b_begin >= s.begin:
                        # ... keep b if after or exactly begin.
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
            self.get_work_dir(s) / f"{f.name.replace(':', '-')}_cn2.nc"
        )  # replace `:` to avoid with tudelft project drive

        if f_out.exists():
            logger.info(f"Output file {f_out} already exists. Skipping.")
            return

        # We need some vars for Cn2 computation. Add them temporarily if not requested.
        extract_vars = copy.deepcopy(self.extract_vars)
        required_vars = ["TSQ", ("EL_PBL", "bottom_top_stag"), "p", "tk"]
        for rv in required_vars:
            if rv not in extract_vars:
                extract_vars.append(rv)

        # Load wrfout file
        logger.info(f"Post-processing {f.name}...")
        ds = load_wrfout(f, vars_to_extract=extract_vars, dt_warmup=None)  # warmup already discarded on file-level

        # Calculate Cn2 and Ct2
        ct2 = get_ct2_hb15(var_theta=ds["TSQ"], Lm=ds["EL_PBL"])
        cn2 = gladstone_cn2_simple(ct2=ct2, p_hPa=ds["p"] / 100, t_K=ds["tk"])
        ds["ct2"] = ct2
        ds["cn2"] = cn2

        # Select only requested variables (including Cn2 and Ct2)
        extract_vars = [v[0] if isinstance(v, tuple) else v for v in self.extract_vars]  # remove destagger dim
        extract_vars += ["ct2", "cn2"]

        # Ugly special cases
        if "uvmet" in extract_vars:
            extract_vars.remove("uvmet")
            extract_vars.append("u_met")
            extract_vars.append("v_met")
        if "wa" in extract_vars:
            extract_vars.remove("wa")
            extract_vars.append("w")

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
        n_done = len(list(self.get_work_dir(s).glob("wrfout_*.nc")))
        return n_done == n_expected
