from __future__ import annotations
from typing import Dict, Iterator, List

import contextlib
import pathlib

from wrf_massive.base import Stage, Simulation, TPath
from wrf_massive.log import get_logger

logger = get_logger("stages.misc.array")


class StageArray(Stage):
    """Stage that holds an array of sub-stages to be run sequentially.
    Only resources defined at the array level are used. This is useful to execute multiple stages in one SLURM job.

    Specifying a `tmp_work_root` runs the SUBSTAGES on temporary storage, e.g. a scratch disk. It does not
    apply to the array's own work dir, which defaults to the simulation dir and must never be moved.
    All substages are moved to tmp up front and moved back only once ALL of them have run, so a later
    substage can consume an earlier one's output while both are still on fast storage. Per-substage
    behaviour (`tmp_teardown_globs`, `tmp_skip_teardown`) is configured on the substages themselves.
    """

    stages: Dict[str, Stage]  # List of stages to run in sequence
    work_dir: TPath = pathlib.Path(".")  # defaults to sim_dir.

    def _propagate_resources(self) -> None:
        """Overwrite substage resources with the array's own, since the array is submitted as one job."""
        for stage in self.stages.values():
            if self.resources != stage.resources:
                logger.warning("Stage array and substage have different resources! Using array resources.")
                logger.warning(f"Overwriting {stage.resources} -> {self.resources}")
                stage.resources = self.resources

    @contextlib.contextmanager
    def tmp_work_dir(
        self,
        s: Simulation,
        *,
        root: pathlib.Path | None = None,
        teardown_globs: List[str] | None = None,
        skip_teardown: bool | None = None,
    ) -> Iterator[pathlib.Path | None]:
        """Run all substage work dirs on temporary storage, tearing them all down only at the very end."""
        root = root if root is not None else self.tmp_work_root

        # A root set on the array applies to all substages; otherwise each substage may bring its own
        to_relocate = {name: (root or stage.tmp_work_root) for name, stage in self.stages.items()}
        to_relocate = {name: sub_root for name, sub_root in to_relocate.items() if sub_root is not None}
        if not to_relocate:
            yield None
            return

        self._propagate_resources()  # teardown parallelism reads substage resources
        with contextlib.ExitStack() as stack:
            for name, sub_root in to_relocate.items():
                stage = self.stages[name]
                if not stage.tmp_work_dir_allowed(s):
                    logger.warning(
                        f"Substage '{name}' ({stage.name}) cannot be run in a tmp work dir "
                        f"(work_dir='{stage.work_dir}'). Leaving it in place."
                    )
                    continue
                stack.enter_context(stage.tmp_work_dir(s, root=sub_root))
            yield None
        # ExitStack unwinds here, in reverse order: teardown of every substage happens only after all
        # of them have run. It also unwinds correctly if a substage raises.

    def teardown_tmp_work_dir(self, s: Simulation, *, all_files: bool = False) -> bool:
        """Force all substage work dirs back from temporary storage.

        The array's own work dir is never relocated (it defaults to the sim dir), so this recurses into
        the substages instead of doing anything to itself.
        """
        self._propagate_resources()  # teardown parallelism reads substage resources

        moved = False
        for name, stage in self.stages.items():
            logger.info(f"Tearing down substage {name} ({stage.name})")
            moved |= stage.teardown_tmp_work_dir(s, all_files=all_files)
        return moved

    def setup(self, s: Simulation):
        self._propagate_resources()

        for name, stage in self.stages.items():
            logger.info(f"Substage is {name} ({stage.name})")
            if not stage.is_setup(s):
                stage.setup(s)
            else:
                logger.info(f"Substage '{stage.name}' already set up. Skipping...")

        logger.info("All substages set up.")

    def is_setup(self, s: Simulation) -> bool:
        # No symlink check needed: the driver enters `tmp_work_dir` before calling this, so the
        # substage work dirs are already symlinked into tmp by the time we get here.
        return all([stage.is_setup(s) for stage in self.stages.values()])

    def run(self, s: Simulation):
        # Run all stages in sequence
        for name, stage in self.stages.items():
            if not stage.is_done(s):
                stage.run(s)
            else:
                logger.info(f"Substage '{stage.name}' already done. Skipping...")
        logger.info("All substages run.")

    def is_done(self, s: Simulation) -> bool:
        return all([stage.is_done(s) for stage in self.stages.values()])
