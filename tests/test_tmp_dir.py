# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Edge cases of the low-level tmp dir helpers. The normal setup/run/teardown roundtrip is covered through the
pipeline in `test_stage_tmp.py`."""

import pytest
from fixtures import StageA, simple_simulation

from wrf_massive.base import Resources, Simulation
from wrf_massive.tmp_dir import setup_tmp_work_dir, teardown_tmp_work_dir, tmp_sim_dir


def _sim(sim_dir) -> Simulation:
    return Simulation(sim_dir=sim_dir, settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02")


def test_tmp_sim_dir_prefixed_with_workspace(tmp_path):
    """Sims of the same name in different workspaces sharing one scratch root must not clash."""
    scratch = tmp_path / "scratch"
    assert [tmp_sim_dir(scratch, _sim(tmp_path / ws / "test_1")) for ws in ("alps", "baltics")] == [
        scratch / "alps" / "test_1",
        scratch / "baltics" / "test_1",
    ]


def test_setup_moves_existing_content(tmp_path, simple_simulation):
    """An existing work dir is moved to tmp and replaced by a symlink."""
    stage_a = StageA(work_dir="a")
    work_dir_orig = stage_a.get_work_dir(simple_simulation)
    (work_dir_orig / "old_file.txt").touch()

    tmp_work_dir = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)

    assert (tmp_work_dir / "old_file.txt").exists()
    assert work_dir_orig.is_symlink()
    assert work_dir_orig.resolve() == tmp_work_dir


@pytest.mark.parametrize("resources", [None, Resources(n_tasks=1, cpus_per_task=4, mem_per_cpu="1G")])
def test_teardown_move_globs(tmp_path, simple_simulation, resources):
    """Only files matching move_globs are moved back, both serially and with a process pool (cpus_total > 1)."""
    stage_a = StageA(work_dir="a", resources=resources)
    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    stage_a.setup(simple_simulation)  # creates setup.txt
    stage_a.run(simple_simulation)  # creates result.txt
    results = ["result.txt"] + [f"result_{i:02d}.txt" for i in range(20)]
    for name in results[1:]:
        (work_dir_tmp / name).touch()

    teardown_tmp_work_dir(simple_simulation, stage_a, move_globs=["result*.txt"])

    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    assert not work_dir_orig.is_symlink()
    assert (work_dir_tmp / "setup.txt").exists()
    assert not (work_dir_orig / "setup.txt").exists()
    for name in results:
        assert (work_dir_orig / name).exists()
        assert not (work_dir_tmp / name).exists()


def test_both_exist(tmp_path, simple_simulation):
    """If original dir and tmp dir both exist, combine them to allow proper running of stages."""
    stage_a = StageA(work_dir="a")
    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    work_dir_orig = stage_a.get_work_dir(simple_simulation)
    stage_a.setup(simple_simulation)  # creates setup.txt
    stage_a.run(simple_simulation)  # creates result.txt

    # Teardown with glob, which leaves setup.txt behind in tmp
    teardown_tmp_work_dir(simple_simulation, stage_a, move_globs=["result*.txt"])

    # Also create subdir with file in original dir to test nested merging
    subdir = work_dir_orig / "subdir"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "file.txt").touch()

    # Now setup again, which should merge the two dirs
    _ = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    assert work_dir_orig.is_symlink()
    assert (work_dir_tmp / "setup.txt").exists()
    assert (work_dir_tmp / "result.txt").exists()
    assert (work_dir_tmp / "subdir" / "file.txt").exists()


def test_setup_dangling_symlink(tmp_path, simple_simulation):
    """A symlink into a wiped tmp root is repaired instead of returning a broken path."""
    stage_a = StageA(work_dir="a")

    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    work_dir_orig.parent.mkdir(parents=True, exist_ok=True)
    work_dir_orig.symlink_to(tmp_path / "gone")
    assert work_dir_orig.is_symlink() and not work_dir_orig.exists()

    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)

    assert work_dir_tmp.exists()
    assert work_dir_orig.resolve() == work_dir_tmp
    stage_a.setup(simple_simulation)
    assert (work_dir_tmp / "setup.txt").exists()


def test_setup_symlink_to_other_target(tmp_path, simple_simulation):
    """If the work dir is symlinked elsewhere, that target is returned so the caller tears down the right dir."""
    stage_a = StageA(work_dir="a")

    other_target = tmp_path / "somewhere_else"
    other_target.mkdir()
    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    work_dir_orig.parent.mkdir(parents=True, exist_ok=True)
    work_dir_orig.symlink_to(other_target)

    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)

    assert work_dir_tmp == other_target
    assert work_dir_orig.resolve() == other_target


def test_teardown_orig_not_symlink(tmp_path, simple_simulation):
    """If the original work dir is a real dir, teardown must abort without clobbering either side."""
    stage_a = StageA(work_dir="a")

    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    stage_a.setup(simple_simulation)

    # Replace the symlink with a real dir, simulating an inconsistent state
    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    work_dir_orig.unlink()
    work_dir_orig.mkdir()
    (work_dir_orig / "local.txt").touch()

    with pytest.raises(RuntimeError, match="to be a symlink"):
        teardown_tmp_work_dir(simple_simulation, stage_a)

    # Neither the original dir nor the results in tmp were touched
    assert (work_dir_orig / "local.txt").exists()
    assert (work_dir_tmp / "setup.txt").exists()


def test_teardown_tmp_dir_gone(tmp_path, simple_simulation):
    """If the tmp dir vanished, teardown drops the dangling symlink instead of raising."""
    stage_a = StageA(work_dir="a")

    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    work_dir_tmp.rmdir()

    teardown_tmp_work_dir(simple_simulation, stage_a)

    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    assert not work_dir_orig.is_symlink()
    assert not work_dir_orig.exists()
