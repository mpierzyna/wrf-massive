# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for `wrf-massive init` and the packaged workspace template."""

import os
import pathlib

import pytest
import yaml
from click.testing import CliRunner

from test_cli import isolate_workspace_state  # noqa: F401 (autouse fixture)
from wrf_massive.cli import main

EXPECTED_FILES = {
    ".gitignore",
    "env.yaml",
    "myoutfields.txt",
    "namelist.tmpl.input",
    "namelist.tmpl.wps",
    "pipeline.py",
    "simulations.py",
    "slurm.job.sh",
}


def invoke(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


@pytest.fixture()
def new_ws(tmp_path) -> pathlib.Path:
    root = tmp_path / "new_ws"
    os.chdir(tmp_path)
    result = invoke("init", root)
    assert result.exit_code == 0, result.output
    return root


def configure_paths(root: pathlib.Path, tmp_path: pathlib.Path):
    """Point the placeholder paths of the template's env.yaml to existing dirs."""
    env = yaml.safe_load((root / "env.yaml").read_text())
    for group, key in [("wps", "wps_tmpl_dir"), ("wps", "geog_data_path"), ("wrf", "wrf_tmpl_dir")]:
        path = tmp_path / key
        path.mkdir()
        env[group][key] = str(path)
    (root / "env.yaml").write_text(yaml.safe_dump(env))


def test_init_creates_template_files(new_ws):
    assert {p.name for p in new_ws.iterdir()} == EXPECTED_FILES
    env = yaml.safe_load((new_ws / "env.yaml").read_text())
    assert env["pipeline"] == "p_default"
    assert env["log_level"] == "info"


def test_init_refuses_to_overwrite_without_force(new_ws):
    (new_ws / "pipeline.py").write_text("# mine")

    result = invoke("init", new_ws)
    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output
    assert (new_ws / "pipeline.py").read_text() == "# mine"

    result = invoke("init", "--force", new_ws)
    assert result.exit_code == 0, result.output
    assert (new_ws / "pipeline.py").read_text() != "# mine"


def test_template_workspace_loads(new_ws, tmp_path):
    """The template's simulations and pipeline import and validate once the env paths are set."""
    configure_paths(new_ws, tmp_path)
    os.chdir(new_ws)

    result = invoke("init-sims", "simulations.py", "sim_test")
    assert result.exit_code == 0, result.output
    assert (new_ws / "test_1" / "simulation.yaml").exists()

    result = invoke("--debug", "submit", "test_1")  # job script from `job_file` in env.yaml
    assert result.exit_code == 0, result.output
    assert result.output.count("Dry run") == 5  # cerra, era5, wps, wrf, sim_done


def test_template_placeholder_paths_give_clear_error(new_ws):
    """Before the env paths are set, commands fail with a pointer to the env file instead of a traceback."""
    os.chdir(new_ws)
    assert invoke("init-sims", "simulations.py", "sim_test").exit_code == 0  # needs no pipeline

    result = invoke("run", "test_1")

    assert result.exit_code != 0
    assert "Failed to load" in result.output
    assert "env.yaml" in result.output
