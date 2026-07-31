"""Tests for the pipeline CLI (`get_pipeline_cli`)."""

import pathlib

import pytest
from click.testing import CliRunner
from fixtures import StageA, simple_simulation

from wrf_massive.base import Pipeline, Simulation
from wrf_massive.cli import get_pipeline_cli


@pytest.fixture()
def tmp_root(tmp_path_factory) -> pathlib.Path:
    """Temporary storage root, separate from the simulation dir."""
    return tmp_path_factory.mktemp("scratch")


@pytest.fixture()
def sim_on_disk(simple_simulation: Simulation) -> Simulation:
    """Simulation with a `simulation.yaml`, so the CLI can load it via `Simulation.from_disk`."""
    simple_simulation.to_disk(simple_simulation.sim_dir.parent)
    return simple_simulation


def test_teardown_requires_sim_dir(sim_on_disk: Simulation):
    """Without a simulation dir the command is a usage error."""
    cli = get_pipeline_cli(Pipeline(a=StageA(work_dir="a")))

    result = CliRunner().invoke(cli, ["teardown"])

    assert result.exit_code != 0
    assert "At least one simulation directory" in result.output


def test_teardown_unknown_stage(sim_on_disk: Simulation):
    """An unknown stage name is rejected before anything is touched."""
    cli = get_pipeline_cli(Pipeline(a=StageA(work_dir="a")))

    result = CliRunner().invoke(cli, ["teardown", "-s", "bogus", str(sim_on_disk.sim_dir)])

    assert result.exit_code != 0
    assert "not found in pipeline" in result.output


def test_teardown_selected_stage(sim_on_disk: Simulation, tmp_root):
    """`-s` tears down only the selected stage; the others stay on tmp storage."""
    stage_a = StageA(work_dir="a", tmp_work_root=tmp_root, tmp_skip_teardown=True)
    stage_b = StageA(work_dir="b", tmp_work_root=tmp_root, tmp_skip_teardown=True)
    p = Pipeline(a=stage_a, b=stage_b)
    p.run(sim_on_disk)
    assert stage_a.get_work_dir(sim_on_disk).is_symlink()
    assert stage_b.get_work_dir(sim_on_disk).is_symlink()

    result = CliRunner().invoke(get_pipeline_cli(p), ["teardown", "-s", "a", str(sim_on_disk.sim_dir)])

    assert result.exit_code == 0, result.output
    assert not stage_a.get_work_dir(sim_on_disk).is_symlink()
    assert (stage_a.get_work_dir(sim_on_disk) / "result.txt").exists()
    assert stage_b.get_work_dir(sim_on_disk).is_symlink()


def test_teardown_all_stages(sim_on_disk: Simulation, tmp_root):
    """Without `-s` every stage is torn down, including ones that were never relocated."""
    stage_a = StageA(work_dir="a", tmp_work_root=tmp_root, tmp_skip_teardown=True)
    stage_b = StageA(work_dir="b")  # never on tmp storage -> no-op, must not fail
    p = Pipeline(a=stage_a, b=stage_b)
    p.run(sim_on_disk)

    result = CliRunner().invoke(get_pipeline_cli(p), ["teardown", str(sim_on_disk.sim_dir)])

    assert result.exit_code == 0, result.output
    assert not stage_a.get_work_dir(sim_on_disk).is_symlink()
    assert (stage_a.get_work_dir(sim_on_disk) / "result.txt").exists()


def test_teardown_all_files_flag(sim_on_disk: Simulation, tmp_root):
    """`--all` overrides the stage's teardown globs."""
    stage = StageA(work_dir="a", tmp_work_root=tmp_root, tmp_teardown_globs=["result*"], tmp_skip_teardown=True)
    p = Pipeline(a=stage)
    p.run(sim_on_disk)

    result = CliRunner().invoke(get_pipeline_cli(p), ["teardown", "--all", str(sim_on_disk.sim_dir)])

    assert result.exit_code == 0, result.output
    work_dir = stage.get_work_dir(sim_on_disk)
    assert (work_dir / "result.txt").exists()
    assert (work_dir / "setup.txt").exists()  # would have been left behind by the globs
