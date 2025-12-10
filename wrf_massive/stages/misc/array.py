from __future__ import annotations
from typing import Dict, List

import pathlib

from wrf_massive.stages.tmp_dir import setup_tmp_work_dir, teardown_tmp_work_dir
from wrf_massive.base import Stage, Simulation, TPath, TPathExists
from wrf_massive.log import get_logger

logger = get_logger("stages.misc.array")


class StageArray(Stage):
    """Stage that holds an array of sub-stages to be run sequentially.
    Only resources defined at the array level are used. This is useful to execute multiple stages in one SLURM job.
    Specifying a `tmp_work_root` allows to run the stages, e.g., on a scratch disk.
    Specifying `stage_tmp_teardown_globs` allows to move back only specific files after all stages are done.
    """

    stages: Dict[str, Stage]  # List of stages to run in sequence
    work_dir: TPath = pathlib.Path(".")  # defaults to sim_dir.
    tmp_work_root: TPathExists | None = None
    stage_tmp_teardown_globs: Dict[str, List[str]] = {}  # glob patterns to teardown tmp dirs per stage

    def setup(self, s: Simulation):

        for name, stage in self.stages.items():
            logger.info(f"Substage is {name} ({stage.name})")

            # Adjust ressources
            if self.resources != stage.resources:
                logger.warning(f"Stage array and substage have different resources! Using array resources.")
                logger.warning(f"Overwriting {stage.resources} -> {self.resources}")
                stage.resources = self.resources

            # If tmp_work_root defined, execute array in tmp dir
            if self.tmp_work_root is not None:
                _ = setup_tmp_work_dir(tmp_root=self.tmp_work_root, stage=stage, s=s)

            # Setup stage
            if not stage.is_setup(s):
                stage.setup(s)
            else:
                logger.info(f"Substage '{stage.name}' already set up. Skipping...")

        logger.info("All substages set up.")

    def is_setup(self, s: Simulation) -> bool:
        def _is_stage_setup(stage: Stage) -> bool:
            """Check if stage is setup. If tmp_work_root is defined, also check if work dir is symlinked."""
            work_dir = stage.get_work_dir(s, create=False)
            if (self.tmp_work_root is not None) and work_dir.exists():
                # We have existing work dir, so check if it is symlinked (to tmp dir)
                # I assume here that checking for a symlink is enough. I don't check
                # if it is linked to a different tmp dir.
                return work_dir.is_symlink() and stage.is_setup(s)
            return stage.is_setup(s)

        return all([_is_stage_setup(stage) for stage in self.stages.values()])

    def run(self, s: Simulation):
        # Run all stages in sequence
        for name, stage in self.stages.items():
            if not stage.is_done(s):
                stage.run(s)
            else:
                logger.info(f"Substage '{stage.name}' already done. Skipping...")
        logger.info("All substages run.")

        # Move back from tmp dir if needed
        if self.tmp_work_root is not None:
            for name, stage in self.stages.items():
                globs = self.stage_tmp_teardown_globs.get(name, None)
                teardown_tmp_work_dir(s=s, stage=stage, move_globs=globs)
            logger.info("All substages torn down from tmp dirs.")

    def is_done(self, s: Simulation) -> bool:
        return all([stage.is_done(s) for stage in self.stages.values()])
