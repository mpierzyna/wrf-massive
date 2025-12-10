from fixtures import StageA, simple_simulation

from wrf_massive.base import Resources
from wrf_massive.stages.tmp_dir import setup_tmp_work_dir, teardown_tmp_work_dir


def test_setup(tmp_path, simple_simulation):
    """Test normal operation: no existing work dir, create tmp work dir and symlink."""
    stage_a = StageA(work_dir="a")

    # Setup tmp work dir
    tmp_work_dir = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    assert tmp_work_dir.exists()

    # Check that symlink is correct
    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    assert work_dir_orig.is_symlink()
    assert work_dir_orig.resolve() == tmp_work_dir

    # Check that stage setup indeed happens in tmp dir
    stage_a.setup(simple_simulation)
    assert (tmp_work_dir / "setup.txt").exists()


def test_setup_existing(tmp_path, simple_simulation):
    """Test operation when existing work dir is present: move to tmp and symlink."""
    stage_a = StageA(work_dir="a")

    # Create existing work dir with some content
    work_dir_orig = stage_a.get_work_dir(simple_simulation)
    work_dir_orig.mkdir(parents=True, exist_ok=True)
    (work_dir_orig / "old_file.txt").touch()

    # Setup tmp work dir
    tmp_work_dir = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    assert tmp_work_dir.exists()

    # Check that existing content was moved to tmp dir
    assert (tmp_work_dir / "old_file.txt").exists()

    # Check that symlink is correct
    assert work_dir_orig.is_symlink()
    assert work_dir_orig.resolve() == tmp_work_dir

    # Check that stage setup indeed happens in tmp dir
    stage_a.setup(simple_simulation)
    assert (tmp_work_dir / "setup.txt").exists()


def test_teardown(tmp_path, simple_simulation):
    """Test normal operation: move results back to original location."""
    stage_a = StageA(work_dir="a")

    # Setup tmp work dir
    _ = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)

    # Perform setup and run in tmp dir
    stage_a.setup(simple_simulation)
    stage_a.run(simple_simulation)

    # Teardown tmp work dir
    teardown_tmp_work_dir(simple_simulation, stage_a)

    # Check that symlink is removed
    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    assert not work_dir_orig.is_symlink()

    # Check that results are moved back to original location
    assert (work_dir_orig / "setup.txt").exists()
    assert (work_dir_orig / "result.txt").exists()


def test_teardown_move_globs(tmp_path, simple_simulation):
    """Test operation with move_globs: only move matching files back to original location."""
    stage_a = StageA(work_dir="a")

    # Setup tmp work dir
    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)

    # Perform setup and run in tmp dir
    stage_a.setup(simple_simulation)  # creates setup.txt
    stage_a.run(simple_simulation)  # creates result.txt

    # Teardown tmp work dir but move only result*.txt files
    teardown_tmp_work_dir(simple_simulation, stage_a, move_globs=["result*.txt"])

    # Check that symlink is removed
    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    assert not work_dir_orig.is_symlink()

    # Check that only matching results are moved back to original location
    assert (work_dir_tmp / "setup.txt").exists()
    assert not (work_dir_orig / "setup.txt").exists()
    assert (work_dir_orig / "result.txt").exists()
    assert not (work_dir_tmp / "result.txt").exists()


def test_teardown_move_globs_mp(tmp_path, simple_simulation):
    """Test operation with move_globs: only move matching files back to original location. (multiprocessing)"""
    stage_a = StageA(work_dir="a", resources=Resources(n_tasks=1, cpus_per_task=4, mem_per_cpu="1G"))

    # Setup tmp work dir
    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)

    # Perform setup and run in tmp dir
    stage_a.setup(simple_simulation)  # creates setup.txt
    stage_a.run(simple_simulation)  # creates result.txt

    # Create some more result files
    for i in range(20):
        (work_dir_tmp / f"result_{i:02d}.txt").touch()

    # Teardown tmp work dir but move only result*.txt files
    teardown_tmp_work_dir(simple_simulation, stage_a, move_globs=["result*.txt"])

    # Check that symlink is removed
    work_dir_orig = stage_a.get_work_dir(simple_simulation, create=False)
    assert not work_dir_orig.is_symlink()

    # Check that only matching results are moved back to original location
    assert (work_dir_tmp / "setup.txt").exists()
    assert not (work_dir_orig / "setup.txt").exists()
    assert (work_dir_orig / "result.txt").exists()
    assert not (work_dir_tmp / "result.txt").exists()

    for i in range(20):
        assert (work_dir_orig / f"result_{i:02d}.txt").exists()
        assert not (work_dir_tmp / f"result_{i:02d}.txt").exists()


def test_both_exist(tmp_path, simple_simulation):
    """If original dir and tmp dir both exist, combine them to allow proper running of stages."""
    stage_a = StageA(work_dir="a", resources=Resources(n_tasks=1, cpus_per_task=4, mem_per_cpu="1G"))

    # Setup tmp work dir
    work_dir_tmp = setup_tmp_work_dir(tmp_path, simple_simulation, stage_a)
    work_dir_orig = stage_a.get_work_dir(simple_simulation)

    # Perform setup and run in tmp dir
    stage_a.setup(simple_simulation)  # creates setup.txt
    stage_a.run(simple_simulation)  # creates result.txt

    # Teardown with glob, which leaves setup.txt behind
    teardown_tmp_work_dir(simple_simulation, stage_a, move_globs=["result*.txt"])
    assert not (work_dir_orig / "setup.txt").exists()
    assert (work_dir_tmp / "setup.txt").exists()
    assert (work_dir_orig / "result.txt").exists()
    assert not (work_dir_tmp / "result.txt").exists()

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
