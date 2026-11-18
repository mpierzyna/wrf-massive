# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the memory/disk monitor (`monitor.sh`, started by `wrf-massive run` via $WRF_MASSIVE_MONITOR)."""

import csv
import os
import pathlib

import pytest
from click.testing import CliRunner

from test_cli import isolate_workspace_state, write_env, ws  # noqa: F401 (fixtures)
from wrf_massive.cli import main
from wrf_massive.monitor import MONITOR_ENVVAR, get_monitor_interval, monitor

MIB = 1024 * 1024


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def run_monitor(tmp_path, work_dir, monkeypatch, **env) -> tuple[list[dict], list[dict]]:
    """Run the monitor around a no-op (first sample + final sample on stop) and return both CSVs."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    prefix = tmp_path / "out" / "stage_1"
    with monitor(60, work_dir, prefix):
        pass
    return read_csv(prefix.with_name("stage_1_mem.csv")), read_csv(prefix.with_name("stage_1_disk.csv"))


def test_node_memory_and_disk_outside_slurm(tmp_path, monkeypatch):
    """Outside SLURM, memory is the node's. Disk usage is measured in the work dir, through a symlink to tmp storage,
    and left empty while the work dir does not exist yet (it is created during setup)."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "data.bin").write_bytes(b"\0" * 3 * MIB)

    mem, disk = run_monitor(tmp_path, work_dir, monkeypatch)

    assert len(mem) == len(disk) == 2  # first sample + final sample when stopped
    assert all(row["source"] == "node" for row in mem)
    assert int(mem[0]["limit_mib"]) > 0 and int(mem[0]["used_mib"]) > 0
    assert int(disk[0]["size_kib"]) >= 3 * 1024

    tmp_dir = tmp_path / "scratch" / "work"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "data.bin").write_bytes(b"\0" * 2 * MIB)
    (tmp_path / "linked").symlink_to(tmp_dir)
    _, disk = run_monitor(tmp_path / "linked_out", tmp_path / "linked", monkeypatch)
    assert int(disk[0]["size_kib"]) >= 2 * 1024

    _, disk = run_monitor(tmp_path / "missing_out", tmp_path / "not_yet", monkeypatch)
    assert disk[0]["size_kib"] == ""


def _write(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_slurm_job_memory(tmp_path, monkeypatch):
    """In a SLURM job, memory comes from the job-level cgroup (all job steps on the node, not the batch step), with
    cgroup v2 or v1, and falls back to the node if there is no job cgroup."""

    def run(case: str, proc_cgroup: str):
        proc = tmp_path / case / "proc_cgroup"
        _write(proc, proc_cgroup)
        env = dict(SLURM_JOB_ID=42, _WRF_MASSIVE_CGROUP_ROOT=tmp_path / case / "cgroup", _WRF_MASSIVE_PROC_CGROUP=proc)
        mem, _ = run_monitor(tmp_path / case, tmp_path, monkeypatch, **env)
        row = mem[0]
        return (row["used_mib"], row["process_mib"], row["peak_mib"], row["limit_mib"]), row["source"]

    root = tmp_path / "v2" / "cgroup"
    _write(root / "cgroup.controllers", "memory\n")
    job = root / "system.slice" / "slurmstepd.scope" / "job_42"
    _write(job / "memory.current", f"{100 * MIB}\n")
    _write(job / "memory.stat", f"anon {50 * MIB}\nfile {40 * MIB}\n")
    _write(job / "memory.peak", f"{200 * MIB}\n")
    _write(job / "memory.max", "max\n")  # = no limit
    _write(job / "step_batch" / "memory.current", f"{1 * MIB}\n")  # must not be used
    v2 = run("v2", "0::/system.slice/slurmstepd.scope/job_42/step_batch/user/task_0\n")
    assert v2 == (("100", "50", "200", ""), "cgroup2")

    job = tmp_path / "v1" / "cgroup" / "memory" / "slurm" / "uid_1000" / "job_42"
    _write(job / "memory.usage_in_bytes", f"{100 * MIB}\n")
    _write(job / "memory.stat", f"rss 0\ntotal_rss {50 * MIB}\n")  # processes sit in the step cgroups below
    _write(job / "memory.max_usage_in_bytes", f"{200 * MIB}\n")
    _write(job / "memory.limit_in_bytes", f"{32 * 1024 * MIB}\n")
    v1 = run("v1", "5:cpuacct,cpu:/slurm/uid_1000/job_42/step_batch\n4:memory:/slurm/uid_1000/job_42/step_batch\n")
    assert v1 == (("100", "50", "200", "32768"), "cgroup1")

    _write(tmp_path / "none" / "cgroup" / "cgroup.controllers", "memory\n")
    assert run("none", "0::/user.slice/user-1000.slice/session-1.scope\n")[1] == "node"


def test_monitor_interval(monkeypatch):
    for value, expected in [("", None), ("30", 30), ("0", ValueError), ("soon", ValueError)]:
        monkeypatch.setenv(MONITOR_ENVVAR, value)
        if expected is ValueError:
            with pytest.raises(ValueError, match=MONITOR_ENVVAR):
                get_monitor_interval()
        else:
            assert get_monitor_interval() == expected, value


def test_run_monitors_each_stage(ws, monkeypatch):
    """With $WRF_MASSIVE_MONITOR, `run` logs one pair of CSVs per stage next to the SLURM output; an invalid interval
    fails before anything runs."""
    result = CliRunner().invoke(main, ["run", "sim_1"], env={MONITOR_ENVVAR: "often"})
    assert result.exit_code != 0
    assert MONITOR_ENVVAR in result.output
    assert not (ws / "sim_1" / "a").exists()

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    os.chdir(ws)
    result = CliRunner().invoke(main, ["run", "sim_1"], env={MONITOR_ENVVAR: "60"})

    assert result.exit_code == 0, result.output
    slurm_dir = ws / "sim_1" / "slurm"
    assert sorted(p.name for p in slurm_dir.iterdir()) == [
        "a_123_disk.csv",
        "a_123_mem.csv",
        "b_123_disk.csv",
        "b_123_mem.csv",
    ]
    assert (ws / "sim_1" / "a" / "result.txt").exists() and (ws / "sim_1" / "b" / "result.txt").exists()
