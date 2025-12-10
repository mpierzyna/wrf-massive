import datetime
import os
import pathlib
import shutil

from wrf_massive.base import Stage, Simulation, TPathExists, TPath
from wrf_massive.log import get_logger
from wrf_massive.stages.tmp_dir import setup_tmp_work_dir, teardown_tmp_work_dir
from wrf_massive.stages.utils import load_wps_wrf_namelist_tmpl, run_cmd_logged

STAGE_DIR = pathlib.Path(os.path.dirname(__file__))

logger = get_logger("stages.wps")


def is_metgrid_successful(wps_dir: pathlib.Path) -> bool:
    """Check if metgrid completed successfully by checking for `met_em*` files and succes in metgrid.log."""

    def _metgrid_log_ok() -> bool:
        """Check that metgrid.log exists and contains success message."""
        metgrid_log = wps_dir / "metgrid.log"
        if not metgrid_log.exists():
            return False
        with metgrid_log.open("r") as f:
            for line in f:
                if "Successful completion of program metgrid.exe" in line:
                    return True
        return False

    def _met_em_files_ok() -> bool:
        """Check that met_em* files exist. We don't know how many there should be, though."""
        met_em_files = list(wps_dir.glob("met_em*"))
        return len(met_em_files) > 0

    return _metgrid_log_ok() and _met_em_files_ok()


class WPSStage(Stage):
    forcing_dir: TPath  # forcing data (grib files), relative to sim_dir
    namelist_tmpl_path: TPathExists
    wps_tmpl_dir: TPathExists  # compiled WPS
    geog_data_path: TPathExists  # WPS geog data dir, absolute path

    def setup(self, s: Simulation):
        def _render_datetime(d: datetime.datetime) -> str:
            """Render datetime in namelist.wps format."""
            return "'" + d.strftime("%Y-%m-%d_%H:%M:%S") + "'"

        # Get working dir
        work_dir = self.get_work_dir(s)

        # Load and render template
        tmpl = load_wps_wrf_namelist_tmpl(self.namelist_tmpl_path)
        for prefix in ["CERRA", "ERA5"]:
            fname = f"namelist.wps.{prefix}"
            tmpl_out = tmpl.render(
                share__start_date=_render_datetime(s.begin_w_warmup),  # begin WITH warmup
                share__end_date=_render_datetime(s.end),
                ungrib__prefix=prefix,
                geogrid__geog_data_path=self.geog_data_path,
            )
            (work_dir / fname).write_text(tmpl_out)
            logger.info(f"-> {fname} rendered.")

        # Copy files
        files = [
            "Vtable.CERRA",  # Copy CERRA Vtable. ERA5 Vtable comes with WPS.
            "setup_wps.sh",
            "run_wps.sh",
            ("gitignore", ".gitignore"),  # rename to .gitignore
        ]
        for src_dst in files:
            if isinstance(src_dst, str):
                src, dst = src_dst, src_dst
            else:
                src, dst = src_dst
            shutil.copy(STAGE_DIR / src, work_dir / dst)
            logger.info(f"-> {dst} copied.")

        # Run setup script
        cmd = ["bash", "setup_wps.sh", self.wps_tmpl_dir]
        run_cmd_logged(cmd, logger=logger, cwd=work_dir, msg="setting up WPS")

    def is_setup(self, s: Simulation) -> bool:
        """Setup if gribfiles linked to WPS dir"""
        return len(list(self.get_work_dir(s).glob("GRIBFILE.*"))) > 0

    def run(self, s: Simulation):
        """Launch run script"""
        cmd = ["bash", "run_wps.sh", s.sim_dir / self.forcing_dir]
        run_cmd_logged(cmd, logger=logger, cwd=self.get_work_dir(s), msg="running WPS")
        logger.info("-> WPS run completed.")

    def is_done(self, s: Simulation) -> bool:
        """Successful if met_em* files exist and log indicates success."""
        return is_metgrid_successful(self.get_work_dir(s))


class WPSTmpDirStage(WPSStage):
    """WPS stage running in temp dir (e.g., ram disk or temp SSD).
    Results are moved back to sim dir after completion."""

    tmp_dir_root: TPathExists  # ramdisk on Linux: /dev/shm

    def setup(self, s: Simulation):
        # Move work dir to tmp location
        _ = setup_tmp_work_dir(tmp_root=self.tmp_dir_root, s=s, stage=self)
        # Continue with normal setup
        super().setup(s)

    def run(self, s: Simulation):
        # Run WPS normally. Because of symlink, this will be in temporary dir.
        super().run(s)

        # Move results back to original work dir
        teardown_tmp_work_dir(s=s, stage=self)
        logger.info(f"-> Done.")
