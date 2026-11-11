# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the `wrf-massive` CLI and workspace discovery."""

import logging
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
from click.testing import CliRunner

from wrf_massive import workspace
from wrf_massive.base import Simulation
from wrf_massive.cli import main
from wrf_massive.workspace import ENV_FILE_ENVVAR, find_env_file

# `fixtures` is importable because pytest puts the tests dir on sys.path.
PIPELINE_PY = """
import pathlib
from fixtures import StageA
from wrf_massive.base import Pipeline, Resources
from wrf_massive.workspace import load_env

env = load_env()
tmp = env.get("tmp_root")
res = Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G")

if env.get("require_marker"):
    assert pathlib.Path("marker.txt").exists(), "relative paths must resolve against the workspace root"

p_test = Pipeline(
    a=StageA(work_dir=env.get("work_dir", "a"), resources=res),
    b=StageA(work_dir="b", resources=res.model_copy(update={"n_nodes": 2})),
)
p_tmp = Pipeline(
    a=StageA(work_dir="a", tmp_work_root=tmp, tmp_skip_teardown=True),
    b=StageA(work_dir="b", tmp_work_root=tmp, tmp_teardown_globs=["result*"], tmp_skip_teardown=True),
)
p_no_resources = Pipeline(a=StageA(work_dir="a"))
"""

SIMULATIONS_PY = """
from wrf_massive.base import Simulation
sim_x = Simulation(sim_dir="sim_x", settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02")
"""


@pytest.fixture(autouse=True)
def isolate_workspace_state(monkeypatch):
    """The CLI changes dir, extends sys.path, imports workspace modules and sets log levels. Undo all of it."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(workspace, "_active", None)
    monkeypatch.delenv(ENV_FILE_ENVVAR, raising=False)
    cwd = os.getcwd()
    levels = {name: logging.getLogger(name).level for name in ("", "wrf_massive")}
    yield
    os.chdir(cwd)
    for name in ("pipeline", "simulations"):
        sys.modules.pop(name, None)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def write_env(path: pathlib.Path, **settings):
    path.write_text("".join(f"{k}: {v}\n" for k, v in settings.items()))


@pytest.fixture()
def ws(tmp_path) -> pathlib.Path:
    """Workspace root with env.yaml, pipeline.py, simulations.py and an initialised simulation `sim_1`."""
    root = tmp_path / "ws"
    root.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    write_env(root / "env.yaml", pipeline="p_test", tmp_root=scratch)
    (root / "pipeline.py").write_text(textwrap.dedent(PIPELINE_PY))
    (root / "simulations.py").write_text(textwrap.dedent(SIMULATIONS_PY))
    Simulation(sim_dir="sim_1", settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02").to_disk(root)
    os.chdir(root)
    return root


def invoke(*args, **kwargs):
    return CliRunner().invoke(main, [str(a) for a in args], **kwargs)


def use_pipeline(ws: pathlib.Path, name: str):
    env = (ws / "env.yaml").read_text().replace("pipeline: p_test", f"pipeline: {name}")
    (ws / "env.yaml").write_text(env)


# Workspace discovery ---------------------------------------------------------------------------------------------


def test_find_env_file_paths_relative_to_cwd(ws):
    """An env file given with directories is resolved against the cwd (bare names are searched upwards)."""
    (ws / "machines").mkdir()
    write_env(ws / "machines" / "env_m.yaml", pipeline="p_test")
    assert find_env_file("machines/env_m.yaml", start=ws) == ws / "machines" / "env_m.yaml"
    assert find_env_file("../machines/env_m.yaml", start=ws / "sim_1") == ws / "machines" / "env_m.yaml"


def test_outside_workspace(tmp_path):
    """--help works anywhere; workspace commands fail with a clear message."""
    os.chdir(tmp_path)
    (tmp_path / "sim").mkdir()
    assert invoke("--help").exit_code == 0
    assert invoke("run", "--help").exit_code == 0

    result = invoke("run", "sim")

    assert result.exit_code != 0
    assert "Env file 'env.yaml' not found" in result.output


def test_run_from_sim_subdir(ws):
    """Run from inside a sim dir: `.` is the sim dir, pipeline.py is found in the root and runs there."""
    write_env(ws / "env.yaml", pipeline="p_test", require_marker=True)
    (ws / "marker.txt").touch()
    os.chdir(ws / "sim_1")

    result = invoke("run", "-s", "a", ".")

    assert result.exit_code == 0, result.output
    assert (ws / "sim_1" / "a" / "result.txt").exists()
    assert not (ws / "sim_1" / "b").exists()
    assert pathlib.Path.cwd() == ws


@pytest.mark.parametrize("via", ["option", "envvar"])
def test_env_file_selection(ws, via):
    """`--env-file`/$WRF_MASSIVE_ENV_FILE (bare name, searched upwards) selects the pipeline AND is what
    `load_env()` returns inside pipeline.py."""
    write_env(ws / "env_alt.yaml", pipeline="p_test", work_dir="alt")
    os.chdir(ws / "sim_1")

    if via == "option":
        result = invoke("--env-file", "env_alt.yaml", "run", "-s", "a", ".")
    else:
        result = invoke("run", "-s", "a", ".", env={ENV_FILE_ENVVAR: "env_alt.yaml"})

    assert result.exit_code == 0, result.output
    assert (ws / "sim_1" / "alt" / "result.txt").exists()
    assert not (ws / "sim_1" / "a").exists()


def test_symlinked_env_file_root_is_link_dir(ws):
    """A symlinked env.yaml makes its own dir the root, not the dir of the link target."""
    (ws / "machines").mkdir()
    (ws / "env.yaml").rename(ws / "machines" / "env_m.yaml")
    (ws / "env.yaml").symlink_to("machines/env_m.yaml")

    result = invoke("run", "-s", "a", "sim_1")

    assert result.exit_code == 0, result.output
    assert pathlib.Path.cwd() == ws


def test_unknown_pipeline(ws):
    use_pipeline(ws, "p_bogus")

    result = invoke("run", "sim_1")

    assert result.exit_code != 0
    assert "Pipeline 'p_bogus'" in result.output
    assert "p_test" in result.output  # lists available pipelines


def test_pipeline_import_error_is_reported(ws):
    """Errors while building the pipeline (e.g. invalid paths in the env file) name the env file."""
    write_env(ws / "env.yaml", pipeline="p_test", require_marker=True)  # marker.txt missing

    result = invoke("run", "sim_1")

    assert result.exit_code != 0
    assert "Failed to load" in result.output
    assert str(ws / "env.yaml") in result.output


# Logging ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("log_level, expected", [(None, logging.INFO), ("WarNing", logging.WARNING)])
def test_log_level(ws, log_level, expected):
    """`log_level` from the env file (case-insensitive) sets the root log level; default is info."""
    settings = {"pipeline": "p_test"} | ({"log_level": log_level} if log_level else {})
    write_env(ws / "env.yaml", **settings)
    assert invoke("run", "sim_1").exit_code == 0
    assert logging.getLogger().level == expected


def test_log_level_invalid(ws):
    write_env(ws / "env.yaml", pipeline="p_test", log_level="loud")

    result = invoke("run", "sim_1")

    assert result.exit_code != 0
    assert "Invalid log_level 'loud'" in result.output


def test_init_sims_from_subdir(ws):
    """Sim dirs are created in the workspace root by default, or in `--root` (relative to the cwd)."""
    os.chdir(ws / "sim_1")

    result = invoke("init-sims", "../simulations.py", "sim_x")
    assert result.exit_code == 0, result.output
    assert (ws / "sim_x" / "simulation.yaml").exists()

    os.chdir(ws / "sim_1")  # the first call changed into the workspace root
    result = invoke("init-sims", "--root", ".", "../simulations.py", "sim_x")
    assert result.exit_code == 0, result.output
    assert (ws / "sim_1" / "sim_x" / "simulation.yaml").exists()


@pytest.fixture()
def fake_sbatch(monkeypatch):
    """Record sbatch calls instead of running them."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{100 + len(calls)}\n", stderr="")

    monkeypatch.setattr("wrf_massive.cli.subprocess.run", fake_run)
    return calls


def test_submit_passes_env_file_to_jobs(ws, fake_sbatch):
    write_env(ws / "env_alt.yaml", pipeline="p_test")
    (ws / "job.sh").touch()
    os.chdir(ws / "sim_1")

    result = invoke("-e", "env_alt.yaml", "submit", "-j", "../job.sh", ".")

    assert result.exit_code == 0, result.output
    assert len(fake_sbatch) == 2  # stages a, b
    (cmd_a, kwargs_a), (cmd_b, _) = fake_sbatch
    assert kwargs_a["env"][ENV_FILE_ENVVAR] == str(ws / "env_alt.yaml")
    assert kwargs_a["cwd"] == ws
    assert cmd_a[-3:] == [str(ws / "job.sh"), "a", str(ws / "sim_1")]
    assert "--nodes=1" in cmd_a  # default
    assert "--nodes=2" in cmd_b
    assert "--dependency=afterok:101" in cmd_b


def test_submit_debug_is_dry_run(ws, fake_sbatch):
    (ws / "job.sh").touch()

    result = invoke("--debug", "submit", "-s", "a", "-j", "job.sh", "sim_1")

    assert result.exit_code == 0, result.output
    assert not fake_sbatch
    assert "Dry run" in result.output
    assert f"{ENV_FILE_ENVVAR}={ws / 'env.yaml'}" in result.output
    assert logging.getLogger("wrf_massive").level == logging.DEBUG


@pytest.mark.parametrize(
    "pipeline, args, message",
    [
        ("p_test", ["teardown"], "At least one simulation directory"),
        ("p_test", ["run", "-s", "bogus", "sim_1"], "not found in pipeline"),
        ("p_no_resources", ["submit", "-j", "job.sh", "sim_1"], "does not have resources"),
    ],
)
def test_usage_errors(ws, fake_sbatch, pipeline, args, message):
    """Invalid arguments are rejected before anything is run or submitted."""
    use_pipeline(ws, pipeline)
    (ws / "job.sh").touch()

    result = invoke(*args)

    assert result.exit_code != 0
    assert message in result.output
    assert not fake_sbatch
    assert not (ws / "sim_1" / "a").exists()


def test_teardown(ws):
    """`-s` tears down only the selected stages; `--all` ignores the stages' teardown globs."""
    use_pipeline(ws, "p_tmp")
    sim = ws / "sim_1"
    assert invoke("run", sim).exit_code == 0
    assert (sim / "a").is_symlink() and (sim / "b").is_symlink()

    result = invoke("teardown", "-s", "a", sim)
    assert result.exit_code == 0, result.output
    assert not (sim / "a").is_symlink()
    assert (sim / "a" / "result.txt").exists()
    assert (sim / "b").is_symlink()

    result = invoke("teardown", "--all", sim)
    assert result.exit_code == 0, result.output
    assert not (sim / "b").is_symlink()
    assert (sim / "b" / "setup.txt").exists()  # would have been left behind by b's globs
