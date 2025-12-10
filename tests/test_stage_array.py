import os
import pathlib
from fixtures import StageA, StageB, simple_simulation

from wrf_massive.base import Simulation, Resources
from wrf_massive.stages.misc import StageArray


def test_resource_inherit(simple_simulation: Simulation):
    """Sub stages should inherit resources from StageArray if defined."""
    stage_a = StageA(work_dir="a", resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"))
    stage_b = StageB(work_dir="b", resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"))

    # If none defined, resources should be None
    sa = StageArray(stages={"a": stage_a, "b": stage_b})
    sa.setup(simple_simulation)
    assert sa.resources == stage_a.resources == stage_b.resources
    assert sa.resources is None

    # If defined at array level, sub stages should inherit
    sa = StageArray(
        stages={"a": stage_a, "b": stage_b},
        resources=Resources(n_tasks=2, cpus_per_task=2, mem_per_cpu="2G"),
    )
    sa.setup(simple_simulation)
    assert sa.resources == stage_a.resources == stage_b.resources
    assert stage_a.resources.n_tasks == 2
    assert stage_b.resources.n_tasks == 2


def test_is_setup(simple_simulation: Simulation):
    """is_setup should return True only if all sub stages are setup."""
    stage_a = StageA(work_dir="a")
    stage_b = StageB(work_dir="b")

    sa = StageArray(stages={"a": stage_a, "b": stage_b})
    assert not sa.is_setup(simple_simulation)

    # Stage b is always setup
    stage_a.setup(simple_simulation)
    assert sa.is_setup(simple_simulation)


def test_is_done(simple_simulation: Simulation):
    """is_done should return True only if all sub stages are done."""
    stage_a = StageA(work_dir="a")
    stage_b = StageB(work_dir="b")

    sa = StageArray(stages={"a": stage_a, "b": stage_b})
    assert not sa.is_done(simple_simulation)

    # Stage b is always done
    stage_a.setup(simple_simulation)
    stage_a.run(simple_simulation)
    assert sa.is_done(simple_simulation)


def test_tmp_root(simple_simulation: Simulation, tmp_path):
    """If tmp_work_root is defined, stages should be setup and run in tmp dir."""
    stage_a = StageA(work_dir="a")
    stage_b = StageB(work_dir="b")

    sa = StageArray(stages={"a": stage_a, "b": stage_b}, tmp_work_root=tmp_path)
    sa.setup(simple_simulation)

    # Check that work dirs are in tmp path
    assert stage_a.get_work_dir(simple_simulation).is_symlink()
    assert stage_b.get_work_dir(simple_simulation).is_symlink()

    sa.run(simple_simulation)

    # After run, work dirs should be back to original
    assert not stage_a.get_work_dir(simple_simulation).is_symlink()
    assert not stage_b.get_work_dir(simple_simulation).is_symlink()


def test_tmp_root_partial_teardown(simple_simulation: Simulation, tmp_path):
    """If tmp_work_root is defined, stages should be setup and run in tmp dir.
    Only specified globs should be moved back.
    """
    stage_a = StageA(work_dir="a")
    stage_b = StageB(work_dir="b")

    sa = StageArray(
        stages={"a": stage_a, "b": stage_b},
        tmp_work_root=tmp_path,
        stage_tmp_teardown_globs={"a": ["result*"]},  # for stage a, move only results back
    )

    sa.setup(simple_simulation)

    # Check that "setup.txt" is created in tmp dir
    stage_a_tmp_dir = pathlib.Path(os.readlink(stage_a.get_work_dir(simple_simulation)))
    assert not stage_a_tmp_dir.is_symlink()
    assert (stage_a_tmp_dir / "setup.txt").exists()

    # Now run and check that only result.txt is moved back
    sa.run(simple_simulation)
    assert (stage_a.get_work_dir(simple_simulation) / "result.txt").exists()
    assert not (stage_a.get_work_dir(simple_simulation) / "setup.txt").exists()
    assert (stage_a_tmp_dir / "setup.txt").exists()


def test_is_setup_not_moved(simple_simulation: Simulation, tmp_path):
    """If all substages are setup but not moved to tmp dir (when requested), is_setup should still return False."""
    stage_a = StageA(work_dir="a")
    stage_b = StageB(work_dir="b")

    sa = StageArray(
        stages={"a": stage_a, "b": stage_b},
        tmp_work_root=tmp_path,
    )

    # Setup stage A manually, which will not move it
    stage_a.setup(simple_simulation)
    assert stage_a.is_setup(simple_simulation)
    assert stage_b.is_setup(simple_simulation)
    assert not stage_a.get_work_dir(simple_simulation).is_symlink()
    assert not stage_b.get_work_dir(simple_simulation).is_symlink()

    # Stage array should still report as NOT setup
    assert not sa.is_setup(simple_simulation)

    # Setting up stage array should now move stages to tmp dir
    sa.setup(simple_simulation)
    assert stage_a.get_work_dir(simple_simulation).is_symlink()
    assert stage_b.get_work_dir(simple_simulation).is_symlink()
    assert sa.is_setup(simple_simulation)

    # Running should work as expected including moving back from tmp dir
    sa.run(simple_simulation)
    assert not stage_a.get_work_dir(simple_simulation).is_symlink()
    assert not stage_b.get_work_dir(simple_simulation).is_symlink()
    assert sa.is_done(simple_simulation)
