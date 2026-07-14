from __future__ import annotations
import datetime
import os
import pathlib
import shutil
from typing import Dict

from wrf_massive.base import Stage, Simulation, TPathExists, TPath
from wrf_massive.log import get_logger
from wrf_massive.stages.utils import load_wps_wrf_namelist_tmpl, run_cmd_logged, get_namelist_value

logger = get_logger("stages.wrf")
STAGE_DIR = pathlib.Path(os.path.dirname(__file__))


def is_wrf_successful(wrf_dir: pathlib.Path) -> bool:
    """WRF finished cleanly if wrfout* files exist AND the rank-0 log reports success.

    `wrfout*` alone is not enough: WRF writes the first history file early, so a run that crashes hours in
    still leaves wrfout files behind. `run_wrf.sh` runs under mpirun, so success is confirmed via the
    `SUCCESS COMPLETE WRF` line in `rsl.out.0000`/`rsl.error.0000`.
    """
    if not any(wrf_dir.glob("wrfout*")):
        return False
    for name in ("rsl.out.0000", "rsl.error.0000"):
        log = wrf_dir / name
        if not log.exists():
            continue
        with log.open("r", errors="ignore") as f:
            if any("SUCCESS COMPLETE WRF" in line for line in f):
                return True
    return False


def _get_time_dict(begin: datetime.datetime, end: datetime.datetime) -> Dict[str, str]:
    return {
        "time__start_year": f"{begin.year}",
        "time__start_month": f"{begin.month:02d}",
        "time__start_day": f"{begin.day:02d}",
        "time__start_hour": f"{begin.hour:02d}",
        "time__start_minute": f"{begin.minute:02d}",
        "time__start_second": "00",
        "time__end_year": f"{end.year}",
        "time__end_month": f"{end.month:02d}",
        "time__end_day": f"{end.day:02d}",
        "time__end_hour": f"{end.hour:02d}",
        "time__end_minute": f"{end.minute:02d}",
        "time__end_second": "00",
    }


class WRFStage(Stage):
    met_em_dir: TPath  # directory with WPS output (met_em files), relative to sim_dir
    wrf_tmpl_dir: TPathExists  # compiled WRF
    namelist_tmpl_path: TPathExists
    myoutfields_path: TPathExists | None

    def setup(self, s: Simulation):
        work_dir = self.get_work_dir(s)

        # Render namelist.input
        tmpl = load_wps_wrf_namelist_tmpl(self.namelist_tmpl_path)
        tmpl_out = tmpl.render(
            **_get_time_dict(s.begin_w_warmup, s.end),  # begin WITH warmup
            **s.settings,
        )
        (work_dir / "namelist.input").write_text(tmpl_out)
        logger.info("-> namelist.input rendered.")

        # Copy files from package dir
        files = [
            "setup_wrf.sh",
            "run_wrf.sh",
            ("gitignore", ".gitignore"),  # rename to .gitignore
        ]
        for src_dst in files:
            if isinstance(src_dst, str):
                src, dst = src_dst, src_dst
            else:
                src, dst = src_dst
            shutil.copy(STAGE_DIR / src, work_dir / dst)
            logger.info(f"-> {dst} copied.")

        # Copy myoutfields.txt from project dir if provided
        if self.myoutfields_path is not None:
            shutil.copy(self.myoutfields_path, work_dir / "myoutfields.txt")
            logger.info("-> myoutfields.txt copied.")

        # Run setup script
        cmd = ["bash", "setup_wrf.sh", s.sim_dir / self.met_em_dir, self.wrf_tmpl_dir]
        run_cmd_logged(cmd, logger=logger, cwd=work_dir, msg="setting up WRF")

    def is_setup(self, s: Simulation) -> bool:
        """Setup if namelist.input exists and met_em* files are linked."""
        return all(
            [
                (self.get_work_dir(s) / "namelist.input").exists(),
                len(list(self.get_work_dir(s).glob("met_em*"))) > 0,
            ]
        )

    def run(self, s: Simulation):
        stage_dir = self.get_work_dir(s)
        # Pass the MPI task count to run_wrf.sh via N_CPUS. Without it, a local (non-SLURM) run falls back to
        # serial mode; under SLURM, SLURM_NTASKS still takes precedence inside the script.
        env = None
        if self.resources is not None:
            env = {**os.environ, "N_CPUS": str(self.resources.n_tasks)}
        run_cmd_logged(["bash", "run_wrf.sh"], cwd=stage_dir, logger=logger, msg="running WRF", env=env)

    def is_done(self, s: Simulation) -> bool:
        """Done only if WRF completed successfully (wrfout* present and rank-0 log reports success)."""
        return is_wrf_successful(self.get_work_dir(s))

    def get_history_interval(self, domain: int, auxhist: int | None = None) -> datetime.timedelta:
        """Get output interval for `domain` (1-indexed!) from namelist.input"""
        field = "history_interval" if auxhist is None else f"auxhist{auxhist}_interval"
        interval_min = get_namelist_value(namelist_path=self.namelist_tmpl_path, field=field)
        if isinstance(interval_min, list):
            interval_min = interval_min[domain - 1]  # domain is 1-indexed
        return datetime.timedelta(minutes=int(interval_min))
