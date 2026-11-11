# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pathlib

import pytest
from fixtures import StageA, StageB, StageFail, StageReadsOther, simple_simulation

from wrf_massive.base import Pipeline, Resources, Simulation
from wrf_massive.stages.misc import MarkDone, StageArray


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


def test_is_setup_and_is_done_require_all_substages(simple_simulation: Simulation):
    stage_a = StageA(work_dir="a")
    stage_b = StageB(work_dir="b")  # always setup and done
    sa = StageArray(stages={"a": stage_a, "b": stage_b})
    assert not sa.is_setup(simple_simulation)
    assert not sa.is_done(simple_simulation)

    stage_a.setup(simple_simulation)
    assert sa.is_setup(simple_simulation)
    assert not sa.is_done(simple_simulation)

    stage_a.run(simple_simulation)
    assert sa.is_done(simple_simulation)


def test_tmp_root_via_pipeline(simple_simulation: Simulation, tmp_path_factory):
    """The production path: a StageArray with tmp_work_root driven by Pipeline.run_stage.

    Substages stay in tmp until ALL of them have run, so a later substage can consume an earlier one's output while
    both are still on fast storage (cf. the production wrf -> cn2 array). Teardown honours each substage's globs, and
    a substage whose work dir IS the sim dir is left in place.
    """
    tmp_root = tmp_path_factory.mktemp("scratch")
    stage_a = StageA(work_dir="a", tmp_teardown_globs=["result*"])
    stage_b = StageReadsOther(work_dir="b", other_work_dir="a")
    stage_done = MarkDone(work_dir=".")

    sa = StageArray(stages={"a": stage_a, "b": stage_b, "done": stage_done}, tmp_work_root=tmp_root)
    p = Pipeline(wrf_cn2=sa)

    p.run_stage(simple_simulation, "wrf_cn2")

    assert stage_b.other_was_symlinked, "substage a was torn down before substage b ran"
    for stage in (stage_a, stage_b):
        assert not stage.get_work_dir(simple_simulation).is_symlink()
        assert (stage.get_work_dir(simple_simulation) / "result.txt").exists()
    assert not (stage_a.get_work_dir(simple_simulation) / "setup.txt").exists()  # not matched by the globs
    assert (tmp_root / simple_simulation.name / "a" / "setup.txt").exists()
    assert not simple_simulation.sim_dir.is_symlink()
    assert (simple_simulation.sim_dir / ".done").exists()
    assert sa.is_done(simple_simulation)

    # Re-running an already-done array must not relocate anything again
    p.run_stage(simple_simulation, "wrf_cn2")
    assert not stage_a.get_work_dir(simple_simulation).is_symlink()


def test_tmp_root_symlinks_before_setup(simple_simulation: Simulation, tmp_path):
    """Entering the array's tmp context moves the substages BEFORE any setup happens, so a substage
    that was set up outside of tmp is picked up and moved along."""
    stage_a = StageA(work_dir="a")
    stage_b = StageB(work_dir="b")

    sa = StageArray(stages={"a": stage_a, "b": stage_b}, tmp_work_root=tmp_path)

    # Setup stage A manually, which will not move it
    stage_a.setup(simple_simulation)
    assert not stage_a.get_work_dir(simple_simulation).is_symlink()

    with sa.tmp_work_dir(simple_simulation):
        # Entering the context symlinks both substages before is_setup/setup are ever called
        assert stage_a.get_work_dir(simple_simulation).is_symlink()
        assert stage_b.get_work_dir(simple_simulation).is_symlink()
        assert sa.is_setup(simple_simulation)

        # Existing setup.txt was carried over into the tmp dir
        stage_a_tmp_dir = pathlib.Path(os.readlink(stage_a.get_work_dir(simple_simulation)))
        assert (stage_a_tmp_dir / "setup.txt").exists()

        sa.setup(simple_simulation)
        sa.run(simple_simulation)

    assert not stage_a.get_work_dir(simple_simulation).is_symlink()
    assert not stage_b.get_work_dir(simple_simulation).is_symlink()
    assert sa.is_done(simple_simulation)


def test_tmp_root_substage_raises(simple_simulation: Simulation, tmp_path):
    """If a substage raises, everything stays in tmp (so a re-run resumes in place) and the error propagates."""
    stage_a = StageA(work_dir="a")
    stage_b = StageFail(work_dir="b")

    sa = StageArray(stages={"a": stage_a, "b": stage_b}, tmp_work_root=tmp_path)

    with pytest.raises(RuntimeError, match="on purpose"):
        with sa.tmp_work_dir(simple_simulation):
            sa.setup(simple_simulation)
            sa.run(simple_simulation)

    assert stage_a.get_work_dir(simple_simulation).is_symlink()
    assert stage_b.get_work_dir(simple_simulation).is_symlink()
    assert (stage_b.get_work_dir(simple_simulation) / "partial.txt").exists()


def test_forced_teardown_recurses_into_substages(simple_simulation: Simulation, tmp_path_factory):
    """An explicit teardown of the array reclaims every substage left on tmp storage."""
    tmp_root = tmp_path_factory.mktemp("scratch")
    stage_a = StageA(work_dir="a", tmp_skip_teardown=True)
    stage_b = StageA(work_dir="b", tmp_skip_teardown=True)
    stage_done = MarkDone(work_dir=".")  # never relocated, must be skipped without error

    sa = StageArray(stages={"a": stage_a, "b": stage_b, "done": stage_done}, tmp_work_root=tmp_root)
    p = Pipeline(wrf_cn2=sa)

    p.run_stage(simple_simulation, "wrf_cn2")
    assert stage_a.get_work_dir(simple_simulation).is_symlink()
    assert stage_b.get_work_dir(simple_simulation).is_symlink()

    p.teardown(simple_simulation)

    assert not stage_a.get_work_dir(simple_simulation).is_symlink()
    assert not stage_b.get_work_dir(simple_simulation).is_symlink()
    assert (stage_a.get_work_dir(simple_simulation) / "result.txt").exists()
    assert (stage_b.get_work_dir(simple_simulation) / "result.txt").exists()
    assert not simple_simulation.sim_dir.is_symlink()


def test_substage_own_tmp_root(simple_simulation: Simulation, tmp_path_factory):
    """A tmp root set on a substage is honoured even when the array itself has none."""
    tmp_root = tmp_path_factory.mktemp("scratch")
    stage_a = StageA(work_dir="a", tmp_work_root=tmp_root)
    stage_b = StageReadsOther(work_dir="b", other_work_dir="a")  # no tmp root -> stays in place

    sa = StageArray(stages={"a": stage_a, "b": stage_b})

    with sa.tmp_work_dir(simple_simulation):
        assert stage_a.get_work_dir(simple_simulation).is_symlink()
        assert not stage_b.get_work_dir(simple_simulation).is_symlink()
        sa.setup(simple_simulation)
        sa.run(simple_simulation)

    assert stage_b.other_was_symlinked, "substage a was torn down before substage b ran"
    assert not stage_a.get_work_dir(simple_simulation).is_symlink()
    assert (stage_a.get_work_dir(simple_simulation) / "result.txt").exists()
