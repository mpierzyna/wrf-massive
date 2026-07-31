"""Tests for running a single Stage on temporary storage (`Stage.tmp_work_dir`)."""

import os
import pathlib
import shutil

import pytest
from fixtures import StageA, StageFail, simple_simulation

from wrf_massive.base import Pipeline, Simulation
from wrf_massive.stages.misc import MarkDone


@pytest.fixture()
def tmp_root(tmp_path_factory) -> pathlib.Path:
    """Temporary storage root, separate from the simulation dir."""
    return tmp_path_factory.mktemp("scratch")


class _RecordingStage(StageA):
    """StageA that records whether its work dir was symlinked while running."""

    was_symlinked: bool = False

    def run(self, s: Simulation):
        self.was_symlinked = self.get_work_dir(s).is_symlink()
        super().run(s)


def test_roundtrip_via_pipeline(simple_simulation: Simulation, tmp_root):
    """A stage with tmp_work_root runs in tmp and has its results moved back, driven by the pipeline."""
    stage = _RecordingStage(work_dir="a", tmp_work_root=tmp_root)
    p = Pipeline(a=stage)

    p.run_stage(simple_simulation, "a")

    assert stage.was_symlinked, "stage did not run in the tmp dir"
    work_dir = stage.get_work_dir(simple_simulation)
    assert not work_dir.is_symlink()
    assert (work_dir / "setup.txt").exists()
    assert (work_dir / "result.txt").exists()
    assert not any((tmp_root / simple_simulation.name).glob("a/*")), "results left behind in tmp"


def test_no_tmp_root_is_noop(simple_simulation: Simulation):
    """Without tmp_work_root, nothing is relocated."""
    stage = _RecordingStage(work_dir="a")
    Pipeline(a=stage).run_stage(simple_simulation, "a")

    assert not stage.was_symlinked
    assert (stage.get_work_dir(simple_simulation) / "result.txt").exists()


def test_failure_leaves_results_in_tmp(simple_simulation: Simulation, tmp_root):
    """If the stage raises, the work dir stays symlinked into tmp and the error propagates."""
    stage = StageFail(work_dir="a", tmp_work_root=tmp_root)
    p = Pipeline(a=stage)

    with pytest.raises(RuntimeError, match="on purpose"):
        p.run_stage(simple_simulation, "a")

    work_dir = stage.get_work_dir(simple_simulation)
    assert work_dir.is_symlink()
    assert pathlib.Path(os.readlink(work_dir)) == tmp_root / simple_simulation.name / "a"
    assert (work_dir / "partial.txt").exists()


def test_rerun_after_failure_resumes_in_place(simple_simulation: Simulation, tmp_root):
    """A re-run picks up the existing tmp dir instead of creating a second one."""
    stage = StageFail(work_dir="a", tmp_work_root=tmp_root)
    with pytest.raises(RuntimeError):
        Pipeline(a=stage).run_stage(simple_simulation, "a")

    work_dir_tmp = pathlib.Path(os.readlink(stage.get_work_dir(simple_simulation)))

    # Second attempt, this time succeeding: same tmp dir, partial output still there
    with stage.tmp_work_dir(simple_simulation) as tmp:
        assert tmp == work_dir_tmp
        assert (stage.get_work_dir(simple_simulation) / "partial.txt").exists()
        (stage.get_work_dir(simple_simulation) / "result.txt").touch()

    work_dir = stage.get_work_dir(simple_simulation)
    assert not work_dir.is_symlink()
    assert (work_dir / "partial.txt").exists()
    assert (work_dir / "result.txt").exists()


def test_skip_teardown(simple_simulation: Simulation, tmp_root):
    """With tmp_skip_teardown, results stay in tmp even after a successful run."""
    stage = StageA(work_dir="a", tmp_work_root=tmp_root, tmp_skip_teardown=True)
    Pipeline(a=stage).run_stage(simple_simulation, "a")

    work_dir = stage.get_work_dir(simple_simulation)
    assert work_dir.is_symlink()
    assert (tmp_root / simple_simulation.name / "a" / "result.txt").exists()


def test_teardown_globs(simple_simulation: Simulation, tmp_root):
    """With tmp_teardown_globs, only matching files are moved back."""
    stage = StageA(work_dir="a", tmp_work_root=tmp_root, tmp_teardown_globs=["result*"])
    Pipeline(a=stage).run_stage(simple_simulation, "a")

    work_dir = stage.get_work_dir(simple_simulation)
    assert not work_dir.is_symlink()
    assert (work_dir / "result.txt").exists()
    assert not (work_dir / "setup.txt").exists()
    assert (tmp_root / simple_simulation.name / "a" / "setup.txt").exists()


def test_refuses_sim_dir_as_work_dir(simple_simulation: Simulation, tmp_root):
    """A stage whose work dir IS the sim dir must never be relocated."""
    stage = MarkDone(work_dir=".", tmp_work_root=tmp_root)
    assert not stage.tmp_work_dir_allowed(simple_simulation)

    with pytest.raises(ValueError, match="simulation dir"):
        with stage.tmp_work_dir(simple_simulation):
            pass

    assert not simple_simulation.sim_dir.is_symlink()


def test_refuses_absolute_work_dir(simple_simulation: Simulation, tmp_root, tmp_path_factory):
    """An absolute work dir cannot be relocated (the tmp path would collapse onto itself)."""
    abs_dir = tmp_path_factory.mktemp("abs_work_dir")
    stage = StageA(work_dir=abs_dir, tmp_work_root=tmp_root)
    assert not stage.tmp_work_dir_allowed(simple_simulation)

    with pytest.raises(ValueError, match="absolute work_dir"):
        with stage.tmp_work_dir(simple_simulation):
            pass


def test_context_is_reentrant(simple_simulation: Simulation, tmp_root):
    """Entering the context twice must not move the work dir a second time."""
    stage = StageA(work_dir="a", tmp_work_root=tmp_root)

    with stage.tmp_work_dir(simple_simulation) as outer:
        assert outer is not None
        with stage.tmp_work_dir(simple_simulation) as inner:
            assert inner is None  # no-op
            stage.setup(simple_simulation)
        # inner exit must NOT have torn down
        assert stage.get_work_dir(simple_simulation).is_symlink()

    assert not stage.get_work_dir(simple_simulation).is_symlink()
    assert (stage.get_work_dir(simple_simulation) / "setup.txt").exists()


def test_get_work_dir_repairs_dangling_symlink(simple_simulation: Simulation, tmp_root):
    """A symlink into a tmp root that has been wiped must not break get_work_dir."""
    stage = StageA(work_dir="a", tmp_work_root=tmp_root)
    work_dir = simple_simulation.sim_dir / "a"
    work_dir.symlink_to(tmp_root / "gone")
    assert work_dir.is_symlink() and not work_dir.exists()

    resolved = stage.get_work_dir(simple_simulation)  # create=True must not raise FileExistsError

    assert resolved.is_dir()
    assert not resolved.is_symlink()


def test_rerun_after_tmp_root_wiped(simple_simulation: Simulation, tmp_root):
    """Full recovery path: run fails, tmp storage is wiped, next run starts clean."""
    stage = StageFail(work_dir="a", tmp_work_root=tmp_root)
    with pytest.raises(RuntimeError):
        Pipeline(a=stage).run_stage(simple_simulation, "a")

    shutil.rmtree(tmp_root / simple_simulation.name)  # scratch cleaned between jobs
    assert stage.get_work_dir(simple_simulation, create=False).is_symlink()

    ok = StageA(work_dir="a", tmp_work_root=tmp_root)
    Pipeline(a=ok).run_stage(simple_simulation, "a")

    work_dir = ok.get_work_dir(simple_simulation)
    assert not work_dir.is_symlink()
    assert (work_dir / "result.txt").exists()


def test_done_marker_skips_before_tmp_setup(simple_simulation: Simulation, tmp_root):
    """A .done marker short-circuits the stage without relocating anything."""
    stage = StageA(work_dir="a", tmp_work_root=tmp_root)
    work_dir = stage.get_work_dir(simple_simulation)
    (work_dir / ".done").touch()

    Pipeline(a=stage).run_stage(simple_simulation, "a")

    assert not work_dir.is_symlink()
    assert not (work_dir / "result.txt").exists()


def test_completed_stage_is_not_relocated(simple_simulation: Simulation, tmp_root):
    """An already setup+done stage is skipped without paying for a move to tmp and back."""
    stage = _RecordingStage(work_dir="a", tmp_work_root=tmp_root)
    Pipeline(a=stage).run_stage(simple_simulation, "a")

    stage.was_symlinked = False
    Pipeline(a=stage).run_stage(simple_simulation, "a")

    assert not stage.was_symlinked, "stage was re-run"
    assert not any((tmp_root / simple_simulation.name).glob("a/*"))
