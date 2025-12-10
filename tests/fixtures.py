import pytest
from wrf_massive.base import Stage, Simulation


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


@pytest.fixture()
def simple_simulation(tmp_path) -> Simulation:
    return Simulation(sim_dir=tmp_path, settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02")
