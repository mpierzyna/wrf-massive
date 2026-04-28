from __future__ import annotations

import os
import pathlib
import re

import pandas as pd

from wrf_massive.base import Stage, Simulation, TPathExists
from wrf_massive.log import get_logger
from wrf_massive.stages.utils import render_template, run_cmd_logged

logger = get_logger("stages.cerra")


STAGE_DIR = pathlib.Path(os.path.dirname(__file__))


def load_cerra_filelist(fpath: pathlib.Path) -> pd.DataFrame:
    """Load CERRA file list from text file into pd.DataFrame.
    Obtained from remote server using `find /path/to/cerra -type f > cerra_filelist.txt`.
    """
    re_timestamp = re.compile(r"\d{4}_\d{2}_\d{2}-\d{2}")

    def _to_series(flist, name) -> pd.Series:
        """Convert list of CERRA filenames to pd.Series with timestamps as index."""
        # To speed up processing of big files, we flatten list first and use regex to find all timestamps at once
        flist_flat = ",".join(flist)
        timestamps = re_timestamp.findall(flist_flat)  #
        timestamps = pd.to_datetime(timestamps, format="%Y_%m_%d-%H")
        return pd.Series(flist, index=timestamps, name=name).sort_index()

    if fpath.suffix == ".gz":
        # gzipped text
        import gzip

        with gzip.open(fpath, "rt") as f:
            files = f.read().splitlines()
    elif fpath.suffix == ".txt":
        # raw text
        files = pathlib.Path(fpath).read_text().splitlines()
    else:
        raise ValueError("File must be .txt or .txt.gz")

    # There are four types: U10_V10, PRES, SFC, soil (from ERA5)
    cerra_pres = [f for f in files if f.endswith("PRES.grb")]
    cerra_uv = [f for f in files if f.endswith("U10_V10.grb")]
    cerra_sfc = [f for f in files if f.endswith("SFC.grb")]
    cerra_soil = [f for f in files if f.endswith("soil.grb")]

    df = pd.concat(
        [
            _to_series(cerra_pres, "PRES"),
            _to_series(cerra_uv, "UV"),
            _to_series(cerra_sfc, "SFC"),
            _to_series(cerra_soil, "SOIL"),
        ],
        axis=1,
    )

    # Ensure no missing files per timestamp
    if df.isnull().any().any():
        missing = df[df.isnull().any(axis=1)].isnull()
        logger.warning(f"Missing CERRA files for timestamps (False = missing):\n{~missing}")
        logger.warning("Records with missing files will be dropped.")
        df = df.dropna()

    return df


class PullCerraStage(Stage):
    remote_flist_path: TPathExists
    remote_path: str
    n_transfers: int = 4
    show_progress: bool = True

    def setup(self, s: Simulation):
        work_dir = self.get_work_dir(s)

        # Render script to pull data and save to stage dir
        rclone_sh = render_template(
            template_path=STAGE_DIR / "pull_cerra_db.tmpl.sh",
            progress="--progress" if self.show_progress else "",
            remote_path=self.remote_path,
            n_transfers=self.n_transfers,
        )
        (work_dir / "pull_cerra.sh").write_text(rclone_sh)
        logger.info(f"-> pull_cerra.sh rendered.")

        # Create include file for rclone pull
        df_inc = load_cerra_filelist(self.remote_flist_path)
        df_inc = df_inc.loc[slice(s.begin_w_warmup, s.end)]

        # Check that we have all expected files in 3h interval
        t_expected = pd.date_range(start=s.begin_w_warmup, end=s.end, freq="3h")
        if not df_inc.index.equals(t_expected):
            missing = t_expected.difference(df_inc.index)
            raise ValueError(f"Missing CERRA files for timestamps:\n{missing}")

        df_inc = df_inc.melt(value_name="path")
        (work_dir / "includes.txt").write_text("\n".join(df_inc["path"].to_list()))
        logger.info(f"-> rclone include file with {len(df_inc)} entries written to includes.txt.")

    def is_setup(self, s: Simulation) -> bool:
        """Assume setup is done when rclone script and `includes.txt` exist."""
        work_dir = self.get_work_dir(s)
        return all(
            [
                (work_dir / "pull_cerra.sh").exists(),
                (work_dir / "includes.txt").exists(),
            ]
        )

    def run(self, s: Simulation):
        work_dir = self.get_work_dir(s)
        cmd = ["bash", "pull_cerra.sh"]
        run_cmd_logged(cmd, logger=logger, cwd=work_dir, msg="pulling CERRA data")
        logger.info("-> Successfully pulled CERRA data.")

    def is_done(self, s: Simulation) -> bool:
        work_dir = self.get_work_dir(s)
        df_inc = pd.read_csv(work_dir / "includes.txt", names=["path"])
        inc_exists = df_inc["path"].apply(lambda f: (work_dir / f).exists())
        logger.debug(inc_exists)
        return inc_exists.all()
