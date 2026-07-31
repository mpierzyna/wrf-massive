import pathlib

import pytest

from wrf_massive.base import Simulation, Stage, TPath

TEST_ROOT = pathlib.Path(__file__).parent
WRFOUT_DIR = TEST_ROOT / "wrfout"


class StageA(Stage):
    """Creates file on setup and run"""

    def is_setup(self, s: Simulation) -> bool:
        return (self.get_work_dir(s) / "setup.txt").exists()

    def setup(self, s: Simulation):
        print("Setting up Stage A")
        work_dir = self.get_work_dir(s)
        (work_dir / "setup.txt").touch()

    def is_done(self, s: Simulation) -> bool:
        return (self.get_work_dir(s) / "result.txt").exists()

    def run(self, s: Simulation):
        print("Running Stage A")
        work_dir = self.get_work_dir(s)
        (work_dir / "result.txt").touch()


class StageB(Stage):
    """Does nothing"""

    def is_setup(self, s: Simulation) -> bool:
        return True

    def setup(self, s: Simulation):
        print("Setting up Stage B")

    def is_done(self, s: Simulation) -> bool:
        return True

    def run(self, s: Simulation):
        print("Running Stage B")


class StageFail(Stage):
    """Creates a file on setup, then produces partial output and raises on run"""

    def is_setup(self, s: Simulation) -> bool:
        return (self.get_work_dir(s) / "setup.txt").exists()

    def setup(self, s: Simulation):
        (self.get_work_dir(s) / "setup.txt").touch()

    def is_done(self, s: Simulation) -> bool:
        return (self.get_work_dir(s) / "result.txt").exists()

    def run(self, s: Simulation):
        (self.get_work_dir(s) / "partial.txt").touch()
        raise RuntimeError("Stage failed on purpose")


class StageReadsOther(Stage):
    """Records at run time whether another stage's work dir is still symlinked (i.e. still in tmp)"""

    other_work_dir: TPath
    other_was_symlinked: bool = False  # set during run()

    def is_setup(self, s: Simulation) -> bool:
        return True

    def setup(self, s: Simulation): ...

    def is_done(self, s: Simulation) -> bool:
        return (self.get_work_dir(s) / "result.txt").exists()

    def run(self, s: Simulation):
        self.other_was_symlinked = (s.sim_dir / self.other_work_dir).is_symlink()
        (self.get_work_dir(s) / "result.txt").touch()


@pytest.fixture()
def simple_simulation(tmp_path) -> Simulation:
    return Simulation(sim_dir=tmp_path, settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02")
