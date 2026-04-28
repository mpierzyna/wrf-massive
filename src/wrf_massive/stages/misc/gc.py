from __future__ import annotations

import shutil
from typing import Callable

from wrf_massive.base import Stage, Simulation
from wrf_massive.log import get_logger

logger = get_logger("stages.misc.gc")


class GarbageCollectStage(Stage):
    """Delete data from `work_dir` (typically previous stage) and mark it for skipping in future runs."""

    armed: bool = False  # False by default. Explicitly arm to enable.
    glob_pattern: str  # Pattern to match files for deletion, e.g. "*.nc"
    run_cond_fn: Callable[[Simulation], bool] | None

    def setup(self, s: Simulation):
        """No setup needed"""

    def is_setup(self, s: Simulation) -> bool:
        """No setup needed"""
        return False  # always return False to avoid "Stage already set up" message

    def run(self, s: Simulation):
        work_dir = self.get_work_dir(s)
        rm_files = sorted(work_dir.glob(self.glob_pattern))
        if not self.armed:
            logger.warning(f"GarbageCollectStage is not armed, skipping deletion of files.")
            logger.info(f"Would delete: {rm_files}.")
            return

        if self.run_cond_fn is not None and not self.run_cond_fn(s):
            logger.warning("Run condition not met, skipping garbage collection.")
            return

        # Delete files matching pattern
        for f in rm_files:
            if f.is_file():
                logger.info(f"-> Deleting file {f}.")
                f.unlink()
            elif f.is_dir():
                logger.info(f"-> Deleting directory {f}.")
                shutil.rmtree(f)

        # Create .done file to mark gc'd stage as done in future runs
        skip_file = work_dir / ".done"
        skip_file.touch()
        logger.info(f"-> Created .done file at to mark stage for skipping.")

    def is_done(self, s: Simulation) -> bool:
        return (self.get_work_dir(s) / ".done").exists()
