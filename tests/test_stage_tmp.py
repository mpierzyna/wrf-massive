# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for running a single Stage on temporary storage (`Stage.tmp_work_dir`)."""

import os
import pathlib
import shutil

import pytest
from fixtures import StageA, StageFail, simple_simulation, tmp_sim_dir

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


def _sim(tmp_path: pathlib.Path, name: str) -> Simulation:
    """A separate simulation per case, so cases in one test cannot interfere."""
    sim_dir = tmp_path / "sims" / name
    sim_dir.mkdir(parents=True)
    return Simulation(sim_dir=sim_dir, settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02")


def test_roundtrip_via_pipeline(tmp_path, tmp_root):
    """A stage with tmp_work_root runs in tmp and has its results (or only those matching the globs) moved back;
    without tmp_work_root nothing is relocated."""
    s = _sim(tmp_path, "no_tmp_root")
    stage = _RecordingStage(work_dir="a")
    Pipeline(a=stage).run_stage(s, "a")
    assert not stage.was_symlinked
    assert (stage.get_work_dir(s) / "result.txt").exists()

    for globs in (None, ["result*"]):
        s = _sim(tmp_path, f"globs_{globs is not None}")
        stage = _RecordingStage(work_dir="a", tmp_work_root=tmp_root, tmp_teardown_globs=globs)

        Pipeline(a=stage).run_stage(s, "a")

        assert stage.was_symlinked, "stage did not run in the tmp dir"
        work_dir = stage.get_work_dir(s)
        tmp_dir = tmp_sim_dir(tmp_root, s) / "a"
        assert not work_dir.is_symlink()
        assert (work_dir / "result.txt").exists()
        if globs is None:
            assert (work_dir / "setup.txt").exists()
            assert not any(tmp_dir.glob("*")), "results left behind in tmp"
        else:
            assert not (work_dir / "setup.txt").exists()
            assert (tmp_dir / "setup.txt").exists()


def test_failure_and_rerun(tmp_path, tmp_root):
    """If the stage raises, the error propagates and the work dir stays symlinked into tmp; a re-run picks up the
    existing tmp dir instead of creating a second one. If tmp storage was wiped in between, the dangling symlink is
    repaired (also by `get_work_dir`, which `is_setup` calls before anything is relocated) and the run starts clean."""
    s = _sim(tmp_path, "resume")
    stage = StageFail(work_dir="a", tmp_work_root=tmp_root)
    with pytest.raises(RuntimeError, match="on purpose"):
        Pipeline(a=stage).run_stage(s, "a")

    work_dir = stage.get_work_dir(s)
    work_dir_tmp = tmp_sim_dir(tmp_root, s) / "a"
    assert work_dir.is_symlink()
    assert pathlib.Path(os.readlink(work_dir)) == work_dir_tmp
    assert (work_dir / "partial.txt").exists()

    # Second attempt, this time succeeding: same tmp dir, partial output still there
    with stage.tmp_work_dir(s) as tmp:
        assert tmp == work_dir_tmp
        assert (work_dir / "partial.txt").exists()
        (work_dir / "result.txt").touch()
    assert not work_dir.is_symlink()
    assert (work_dir / "partial.txt").exists() and (work_dir / "result.txt").exists()

    s = _sim(tmp_path, "wiped")
    with pytest.raises(RuntimeError):
        Pipeline(a=StageFail(work_dir="a", tmp_work_root=tmp_root)).run_stage(s, "a")
    shutil.rmtree(tmp_sim_dir(tmp_root, s))  # scratch cleaned between jobs
    ok = StageA(work_dir="a", tmp_work_root=tmp_root)
    assert ok.get_work_dir(s, create=False).is_symlink()

    Pipeline(a=ok).run_stage(s, "a")

    work_dir = ok.get_work_dir(s)
    assert not work_dir.is_symlink()
    assert (work_dir / "result.txt").exists()


def test_forced_teardown(tmp_path, tmp_root):
    """`Pipeline.teardown` reclaims work dirs left in tmp by `tmp_skip_teardown`, ignoring `.done` markers.

    It honours `tmp_teardown_globs` unless `all_files` is set, and skips stages that were never relocated.
    """
    for all_files in (False, True):
        s = _sim(tmp_path, f"all_files_{all_files}")
        stage = StageA(work_dir="a", tmp_work_root=tmp_root, tmp_teardown_globs=["result*"], tmp_skip_teardown=True)
        never_relocated = StageA(work_dir="b")
        p = Pipeline(a=stage, b=never_relocated)
        p.run(s)

        work_dir = stage.get_work_dir(s)
        tmp_dir = tmp_sim_dir(tmp_root, s) / "a"
        assert work_dir.is_symlink(), "tmp_skip_teardown should leave the work dir in tmp"
        (work_dir / ".done").touch()
        (s.sim_dir / ".done").touch()

        p.teardown(s, all_files=all_files)

        assert not work_dir.is_symlink()
        assert (work_dir / "result.txt").exists()
        assert (work_dir / "setup.txt").exists() == all_files
        assert tmp_dir.exists() != all_files
        assert p.teardown_stage(s, "b") is False
        assert (never_relocated.get_work_dir(s) / "result.txt").exists()


def test_refuses_to_relocate(simple_simulation: Simulation, tmp_root, tmp_path_factory):
    """The sim dir itself (e.g. MarkDone) and absolute work dirs must never be relocated."""
    for work_dir, match in [(".", "simulation dir"), (tmp_path_factory.mktemp("abs_work_dir"), "absolute work_dir")]:
        stage = MarkDone(work_dir=work_dir, tmp_work_root=tmp_root)
        assert not stage.tmp_work_dir_allowed(simple_simulation)

        with pytest.raises(ValueError, match=match):
            with stage.tmp_work_dir(simple_simulation):
                pass

        assert not simple_simulation.sim_dir.is_symlink()


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


def test_finished_stage_is_not_relocated(tmp_path, tmp_root):
    """A stage with a .done marker, or already setup+done, is skipped without a move to tmp and back."""
    for finished_by in ("done_marker", "previous_run"):
        s = _sim(tmp_path, finished_by)
        stage = _RecordingStage(work_dir="a", tmp_work_root=tmp_root)
        work_dir = stage.get_work_dir(s)
        if finished_by == "done_marker":
            (work_dir / ".done").touch()
        else:
            Pipeline(a=stage).run_stage(s, "a")
            stage.was_symlinked = False

        Pipeline(a=stage).run_stage(s, "a")

        assert not stage.was_symlinked, f"{finished_by}: stage was re-run"
        assert not work_dir.is_symlink()
        assert not any(tmp_sim_dir(tmp_root, s).glob("a/*"))
