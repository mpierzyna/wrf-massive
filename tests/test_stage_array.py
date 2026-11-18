# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pathlib

import pytest
from fixtures import StageA, StageB, StageFail, StageReadsOther, simple_simulation, tmp_sim_dir

from wrf_massive.base import Pipeline, Resources, Simulation
from wrf_massive.stages.misc import MarkDone, StageArray


def _sim(tmp_path: pathlib.Path, name: str) -> Simulation:
    """A separate simulation per case, so cases in one test cannot interfere."""
    sim_dir = tmp_path / "sims" / name
    sim_dir.mkdir(parents=True)
    return Simulation(sim_dir=sim_dir, settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02")


def test_resources_and_state(simple_simulation: Simulation):
    """Sub stages inherit resources from the StageArray if defined; the array is setup/done only if all are."""
    res = Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G")
    stage_a, stage_b = StageA(work_dir="a", resources=res), StageB(work_dir="b", resources=res)

    sa = StageArray(stages={"a": stage_a, "b": stage_b})  # none at array level -> None everywhere
    sa.setup(simple_simulation)
    assert sa.resources is stage_a.resources is stage_b.resources is None

    sa = StageArray(
        stages={"a": stage_a, "b": stage_b}, resources=Resources(n_tasks=2, cpus_per_task=2, mem_per_cpu="2G")
    )
    sa.setup(simple_simulation)
    assert sa.resources == stage_a.resources == stage_b.resources
    assert stage_a.resources.n_tasks == stage_b.resources.n_tasks == 2

    stage_a = StageA(work_dir="a2")
    sa = StageArray(stages={"a": stage_a, "b": StageB(work_dir="b")})  # b is always setup and done
    assert not sa.is_setup(simple_simulation) and not sa.is_done(simple_simulation)
    stage_a.setup(simple_simulation)
    assert sa.is_setup(simple_simulation) and not sa.is_done(simple_simulation)
    stage_a.run(simple_simulation)
    assert sa.is_done(simple_simulation)


def test_tmp_root_via_pipeline(tmp_path, tmp_path_factory):
    """The production path: a StageArray with tmp_work_root driven by Pipeline.run_stage.

    Substages stay in tmp until ALL of them have run, so a later substage can consume an earlier one's output while
    both are still on fast storage (cf. the production wrf -> cn2 array). Teardown honours each substage's globs, and
    a substage whose work dir IS the sim dir is left in place. An explicit teardown reclaims every substage left on
    tmp storage by `tmp_skip_teardown`.
    """
    tmp_root = tmp_path_factory.mktemp("scratch")
    s = _sim(tmp_path, "run")
    stage_a = StageA(work_dir="a", tmp_teardown_globs=["result*"])
    stage_b = StageReadsOther(work_dir="b", other_work_dir="a")
    sa = StageArray(stages={"a": stage_a, "b": stage_b, "done": MarkDone(work_dir=".")}, tmp_work_root=tmp_root)
    p = Pipeline(wrf_cn2=sa)

    p.run_stage(s, "wrf_cn2")

    assert stage_b.other_was_symlinked, "substage a was torn down before substage b ran"
    for stage in (stage_a, stage_b):
        assert not stage.get_work_dir(s).is_symlink()
        assert (stage.get_work_dir(s) / "result.txt").exists()
    assert not (stage_a.get_work_dir(s) / "setup.txt").exists()  # not matched by the globs
    assert (tmp_sim_dir(tmp_root, s) / "a" / "setup.txt").exists()
    assert not s.sim_dir.is_symlink()
    assert (s.sim_dir / ".done").exists()
    assert sa.is_done(s)

    # Re-running an already-done array must not relocate anything again
    p.run_stage(s, "wrf_cn2")
    assert not stage_a.get_work_dir(s).is_symlink()

    s = _sim(tmp_path, "forced_teardown")
    stage_a, stage_b = StageA(work_dir="a", tmp_skip_teardown=True), StageA(work_dir="b", tmp_skip_teardown=True)
    stages = {"a": stage_a, "b": stage_b, "done": MarkDone(work_dir=".")}  # done: never relocated, skipped
    p = Pipeline(wrf_cn2=StageArray(stages=stages, tmp_work_root=tmp_root))
    p.run_stage(s, "wrf_cn2")
    assert stage_a.get_work_dir(s).is_symlink() and stage_b.get_work_dir(s).is_symlink()

    p.teardown(s)

    for stage in (stage_a, stage_b):
        assert not stage.get_work_dir(s).is_symlink()
        assert (stage.get_work_dir(s) / "result.txt").exists()
    assert not s.sim_dir.is_symlink()


def test_tmp_context_of_substages(tmp_path, tmp_path_factory):
    """Entering the array's tmp context moves the substages BEFORE any setup happens, so a substage that was set up
    outside of tmp is picked up and moved along. A tmp root set on a substage is honoured even when the array itself
    has none."""
    s = _sim(tmp_path, "array_root")
    stage_a, stage_b = StageA(work_dir="a"), StageB(work_dir="b")
    sa = StageArray(stages={"a": stage_a, "b": stage_b}, tmp_work_root=tmp_path_factory.mktemp("scratch"))
    stage_a.setup(s)  # set up manually, which does not move it
    assert not stage_a.get_work_dir(s).is_symlink()

    with sa.tmp_work_dir(s):
        assert stage_a.get_work_dir(s).is_symlink() and stage_b.get_work_dir(s).is_symlink()
        assert sa.is_setup(s)
        assert (pathlib.Path(os.readlink(stage_a.get_work_dir(s))) / "setup.txt").exists()  # carried over
        sa.setup(s)
        sa.run(s)

    assert not stage_a.get_work_dir(s).is_symlink() and not stage_b.get_work_dir(s).is_symlink()
    assert sa.is_done(s)

    s = _sim(tmp_path, "substage_root")
    stage_a = StageA(work_dir="a", tmp_work_root=tmp_path_factory.mktemp("scratch"))
    stage_b = StageReadsOther(work_dir="b", other_work_dir="a")  # no tmp root -> stays in place
    sa = StageArray(stages={"a": stage_a, "b": stage_b})

    with sa.tmp_work_dir(s):
        assert stage_a.get_work_dir(s).is_symlink()
        assert not stage_b.get_work_dir(s).is_symlink()
        sa.setup(s)
        sa.run(s)

    assert stage_b.other_was_symlinked, "substage a was torn down before substage b ran"
    assert not stage_a.get_work_dir(s).is_symlink()
    assert (stage_a.get_work_dir(s) / "result.txt").exists()


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
