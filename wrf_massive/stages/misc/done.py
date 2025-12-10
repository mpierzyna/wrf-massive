from __future__ import annotations
from typing import Callable

from wrf_massive.base import Stage, Simulation
from wrf_massive.log import get_logger

logger = get_logger("stages.misc.done")


class MarkDone(Stage):
    """Stage that marks its work directory as done by creating a `.done` file."""

    run_cond_fn: Callable[[Simulation], bool] | None = None  # Optional condition to run

    def setup(self, s: Simulation): ...

    def is_setup(self, s: Simulation) -> bool:
        return False  # false to avoid "Stage already set up" message

    def run(self, s: Simulation):
        if self.run_cond_fn is not None:
            try:
                if not self.run_cond_fn(s):
                    logger.warning("Run condition not met, skipping marking as done.")
                    return
            except Exception as e:
                logger.error(f"Error evaluating run condition: {e}. Not marking as done.")
                return

        (self.get_work_dir(s) / ".done").touch()
        logger.info(f"Marked directory {self.get_work_dir(s)} as done.")

    def is_done(self, s: Simulation) -> bool:
        return (self.get_work_dir(s) / ".done").exists()
