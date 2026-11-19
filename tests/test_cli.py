# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the `wrf-massive` CLI and workspace discovery."""

import hashlib
import logging
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
from click.testing import CliRunner

from wrf_massive.base import Simulation
from wrf_massive.cli import main
from wrf_massive.workspace import ENV_HASH_ENVVAR, find_env_file, load_env

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
    b=StageA(work_dir="b", resources=res.model_copy(update={"n_nodes": 2, "partition": "fat"})),
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
    monkeypatch.delenv(ENV_HASH_ENVVAR, raising=False)
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


def test_workspace_discovery(ws):
    """env.yaml is searched upwards (other env files are never used) and its dir is the root, also when it is a
    symlink. `.` inside a sim dir is that sim dir; pipeline.py runs in the root, where `load_env()` sees the env."""
    write_env(ws / "env.yaml", pipeline="p_test", require_marker=True, work_dir="alt")
    (ws / "marker.txt").touch()
    write_env(ws / "sim_1" / "env_m.yaml", pipeline="p_test")
    assert find_env_file(start=ws / "sim_1") == ws / "env.yaml"

    os.chdir(ws / "sim_1")
    assert load_env()["work_dir"] == "alt"
    result = invoke("run", "-s", "a", ".")
    assert result.exit_code == 0, result.output
    assert (ws / "sim_1" / "alt" / "result.txt").exists()
    assert not (ws / "sim_1" / "b").exists()
    assert pathlib.Path.cwd() == ws

    # A symlinked env.yaml makes its own dir the root, not the dir of the link target
    (ws / "machines").mkdir()
    (ws / "env.yaml").rename(ws / "machines" / "env_m.yaml")
    (ws / "env.yaml").symlink_to("machines/env_m.yaml")
    os.chdir(ws / "sim_1")
    result = invoke("run", "-s", "a", ".")
    assert result.exit_code == 0, result.output
    assert pathlib.Path.cwd() == ws


def test_outside_workspace(tmp_path):
    """--help works anywhere; workspace commands fail with a clear message."""
    os.chdir(tmp_path)
    (tmp_path / "sim").mkdir()
    assert invoke("--help").exit_code == 0
    assert invoke("run", "--help").exit_code == 0

    result = invoke("run", "sim")

    assert result.exit_code != 0
    assert "No env.yaml found" in result.output


def test_log_level(ws):
    """`log_level` from the env file (case-insensitive) sets the root log level; default is info."""
    for settings, expected in [({}, logging.INFO), ({"log_level": "WarNing"}, logging.WARNING)]:
        write_env(ws / "env.yaml", pipeline="p_test", **settings)
        assert invoke("run", "sim_1").exit_code == 0
        assert logging.getLogger().level == expected


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


# Errors ----------------------------------------------------------------------------------------------------------


@pytest.fixture()
def fake_sbatch(monkeypatch):
    """Record sbatch calls instead of running them."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{100 + len(calls)}\n", stderr="")

    monkeypatch.setattr("wrf_massive.cli.subprocess.run", fake_run)
    return calls


def env_hash(ws: pathlib.Path) -> str:
    return hashlib.sha256((ws / "env.yaml").read_bytes()).hexdigest()


def exported_env(cmd: list[str]) -> dict[str, str]:
    """The variables an sbatch command passes to the job via `--export=ALL,K=V,...`."""
    (export,) = [a for a in cmd if a.startswith("--export=")]
    names = export.removeprefix("--export=").split(",")
    assert names[0] == "ALL"  # the submitting environment is still exported
    return dict(n.split("=", 1) for n in names[1:])


def test_errors_are_reported_before_anything_runs(ws, fake_sbatch):
    """Invalid env files, pipelines and arguments are rejected with a clear message before anything is run or
    submitted."""
    (ws / "job.sh").touch()
    cases = [  # env.yaml settings, CLI args, expected messages
        ({"pipeline": "p_bogus"}, ["run", "sim_1"], ["Pipeline 'p_bogus'", "p_test"]),  # lists available pipelines
        ({"require_marker": True}, ["run", "sim_1"], ["Failed to load", str(ws / "env.yaml")]),  # pipeline.py raises
        ({"log_level": "loud"}, ["run", "sim_1"], ["Invalid log_level 'loud'"]),
        ({}, ["teardown"], ["At least one simulation directory"]),
        ({}, ["run", "-s", "bogus", "sim_1"], ["not found in pipeline"]),
        ({"pipeline": "p_no_resources"}, ["submit", "-j", "job.sh", "sim_1"], ["does not have resources"]),
        ({}, ["submit", "sim_1"], ["No job script: set `job_file`"]),
        ({"job_file": "missing.sh"}, ["submit", "sim_1"], ["does not exist"]),
    ]
    for settings, args, messages in cases:
        write_env(ws / "env.yaml", **({"pipeline": "p_test"} | settings))
        os.chdir(ws)

        result = invoke(*args)

        assert result.exit_code != 0, args
        for message in messages:
            assert message in result.output, (args, result.output)
        assert not fake_sbatch
        assert not (ws / "sim_1" / "a").exists()


def test_job_fails_if_env_changed_since_submit(ws):
    """A job submitted with one env.yaml fails loudly instead of running with another one."""
    (ws / "env.yaml").rename(ws / "env_m1.yaml")
    (ws / "env.yaml").symlink_to("env_m1.yaml")
    submitted = env_hash(ws)
    original = (ws / "env_m1.yaml").read_text()

    result = invoke("run", "-s", "b", "sim_1", env={ENV_HASH_ENVVAR: submitted})
    assert result.exit_code == 0, result.output  # unchanged

    def relink():
        write_env(ws / "env_m2.yaml", pipeline="p_test", work_dir="alt")
        (ws / "env.yaml").unlink()
        (ws / "env.yaml").symlink_to("env_m2.yaml")

    for change in [lambda: write_env(ws / "env_m1.yaml", pipeline="p_test", work_dir="alt"), relink]:
        change()
        os.chdir(ws)

        result = invoke("run", "sim_1", env={ENV_HASH_ENVVAR: submitted})

        assert result.exit_code != 0
        assert "changed since this job was submitted" in result.output
        assert ("-> env_m2.yaml" in result.output) == (change is relink)
        assert not (ws / "sim_1" / "a").exists() and not (ws / "sim_1" / "alt").exists()
        (ws / "env_m1.yaml").write_text(original)


# Submit and teardown ---------------------------------------------------------------------------------------------


def test_submit(ws, fake_sbatch):
    """Stages are chained with afterok and the env hash; the job file comes from `-j` or else from the env file's
    `job_file` (relative to the workspace root); `--debug` is a dry run."""
    (ws / "job.sh").touch()
    os.chdir(ws / "sim_1")
    result = invoke("submit", "-j", "../job.sh", ".")
    assert result.exit_code == 0, result.output
    assert len(fake_sbatch) == 2  # stages a, b
    (cmd_a, kwargs_a), (cmd_b, _) = fake_sbatch
    assert exported_env(cmd_a) == {ENV_HASH_ENVVAR: env_hash(ws)}
    assert "env" not in kwargs_a  # passed via --export, not via the sbatch process env (dropped on some clusters)
    assert kwargs_a["cwd"] == ws
    assert cmd_a[-3:] == [str(ws / "job.sh"), "a", str(ws / "sim_1")]
    assert "--nodes=1" in cmd_a  # default
    assert "--nodes=2" in cmd_b
    assert not [arg for arg in cmd_a if arg.startswith("--partition")]  # default: keep the job file's partition
    assert "--partition=fat" in cmd_b
    assert "--dependency=afterok:101" in cmd_b

    write_env(ws / "env.yaml", pipeline="p_test", job_file="job.sh")
    (ws / "other.sh").touch()
    os.chdir(ws / "sim_1")
    assert invoke("submit", "-s", "a", ".").exit_code == 0
    assert fake_sbatch[-1][0][-3] == str(ws / "job.sh")
    os.chdir(ws / "sim_1")
    assert invoke("submit", "-s", "a", "-j", "../other.sh", ".").exit_code == 0
    assert fake_sbatch[-1][0][-3] == str(ws / "other.sh")

    n_submitted = len(fake_sbatch)
    result = invoke("--debug", "submit", "-s", "a", "-j", "job.sh", "sim_1")
    assert result.exit_code == 0, result.output
    assert len(fake_sbatch) == n_submitted
    assert "Dry run" in result.output
    assert f"{ENV_HASH_ENVVAR}={env_hash(ws)}" in result.output
    assert logging.getLogger("wrf_massive").level == logging.DEBUG


def test_submit_force_setup(ws, fake_sbatch):
    """With `--force-setup`, the submitted jobs rerun the setup of their stage, without it they don't."""
    (ws / "job.sh").touch()
    assert invoke("run", "-s", "a", "sim_1").exit_code == 0  # set up and done

    def run_job(*submit_args):
        assert invoke("submit", "-s", "a", "-j", "job.sh", *submit_args, "sim_1").exit_code == 0
        cmd, _ = fake_sbatch[-1]
        env = exported_env(cmd)  # set by `submit` for the job
        return invoke("run", f"--stages={cmd[-2]}", cmd[-1], env=env)  # what the job script runs

    result = run_job("--force-setup")
    assert result.exit_code == 0, result.output
    assert "Setting up Stage A" in result.output
    result = run_job()
    assert result.exit_code == 0, result.output
    assert "Setting up Stage A" not in result.output

    result = invoke("--debug", "submit", "-s", "a", "-j", "job.sh", "--force-setup", "sim_1")
    assert result.exit_code == 0, result.output
    assert "FORCE_SETUP=1" in result.output


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
