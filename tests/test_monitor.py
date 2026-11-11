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


def test_node_memory_outside_slurm(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "data.bin").write_bytes(b"\0" * 3 * MIB)

    mem, disk = run_monitor(tmp_path, work_dir, monkeypatch)

    assert len(mem) == len(disk) == 2  # first sample + final sample when stopped
    assert all(row["source"] == "node" for row in mem)
    assert int(mem[0]["limit_mib"]) > 0 and int(mem[0]["used_mib"]) > 0
    assert int(disk[0]["size_kib"]) >= 3 * 1024


def test_disk_follows_symlinked_work_dir(tmp_path, monkeypatch):
    """A stage on temporary storage is a symlink to the tmp dir; its size is measured there."""
    tmp_dir = tmp_path / "scratch" / "work"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "data.bin").write_bytes(b"\0" * 2 * MIB)
    (tmp_path / "work").symlink_to(tmp_dir)

    _, disk = run_monitor(tmp_path, tmp_path / "work", monkeypatch)

    assert int(disk[0]["size_kib"]) >= 2 * 1024


def test_missing_work_dir_gives_empty_size(tmp_path, monkeypatch):
    """The work dir is created during setup, so it may not exist for the first samples."""
    _, disk = run_monitor(tmp_path, tmp_path / "not_yet", monkeypatch)
    assert disk[0]["size_kib"] == ""


def _write(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_slurm_job_cgroup_v2(tmp_path, monkeypatch):
    """Memory of the job-level cgroup (all job steps on the node), not of the batch step or the node."""
    root = tmp_path / "cgroup"
    _write(root / "cgroup.controllers", "memory\n")
    job = root / "system.slice" / "slurmstepd.scope" / "job_42"
    _write(job / "memory.current", f"{100 * MIB}\n")
    _write(job / "memory.stat", f"anon {50 * MIB}\nfile {40 * MIB}\n")
    _write(job / "memory.peak", f"{200 * MIB}\n")
    _write(job / "memory.max", "max\n")
    _write(job / "step_batch" / "memory.current", f"{1 * MIB}\n")  # must not be used
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/system.slice/slurmstepd.scope/job_42/step_batch/user/task_0\n")

    mem, _ = run_monitor(
        tmp_path, tmp_path, monkeypatch, SLURM_JOB_ID=42, _WRF_MASSIVE_CGROUP_ROOT=root, _WRF_MASSIVE_PROC_CGROUP=proc
    )

    assert mem[0] | {"timestamp": "", "elapsed_s": ""} == {
        "timestamp": "",
        "elapsed_s": "",
        "used_mib": "100",
        "process_mib": "50",
        "peak_mib": "200",
        "limit_mib": "",  # "max" = no limit
        "source": "cgroup2",
    }


def test_slurm_job_cgroup_v1(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    job = root / "memory" / "slurm" / "uid_1000" / "job_42"
    _write(job / "memory.usage_in_bytes", f"{100 * MIB}\n")
    _write(job / "memory.stat", f"rss 0\ntotal_rss {50 * MIB}\n")  # processes sit in the step cgroups below
    _write(job / "memory.max_usage_in_bytes", f"{200 * MIB}\n")
    _write(job / "memory.limit_in_bytes", f"{32 * 1024 * MIB}\n")
    proc = tmp_path / "proc_cgroup"
    _write(proc, "5:cpuacct,cpu:/slurm/uid_1000/job_42/step_batch\n4:memory:/slurm/uid_1000/job_42/step_batch\n")

    mem, _ = run_monitor(
        tmp_path, tmp_path, monkeypatch, SLURM_JOB_ID=42, _WRF_MASSIVE_CGROUP_ROOT=root, _WRF_MASSIVE_PROC_CGROUP=proc
    )

    row = mem[0]
    assert (row["used_mib"], row["process_mib"], row["peak_mib"], row["limit_mib"]) == ("100", "50", "200", "32768")
    assert row["source"] == "cgroup1"


def test_slurm_job_without_job_cgroup_falls_back_to_node(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    _write(root / "cgroup.controllers", "memory\n")
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/user.slice/user-1000.slice/session-1.scope\n")

    mem, _ = run_monitor(
        tmp_path, tmp_path, monkeypatch, SLURM_JOB_ID=42, _WRF_MASSIVE_CGROUP_ROOT=root, _WRF_MASSIVE_PROC_CGROUP=proc
    )

    assert mem[0]["source"] == "node"


@pytest.mark.parametrize("value, expected", [("", None), ("30", 30), ("0", ValueError), ("soon", ValueError)])
def test_monitor_interval(monkeypatch, value, expected):
    monkeypatch.setenv(MONITOR_ENVVAR, value)
    if expected is ValueError:
        with pytest.raises(ValueError, match=MONITOR_ENVVAR):
            get_monitor_interval()
    else:
        assert get_monitor_interval() == expected


def test_run_monitors_each_stage(ws, monkeypatch):
    """With $WRF_MASSIVE_MONITOR, `run` logs one pair of CSVs per stage next to the SLURM output."""
    monkeypatch.setenv("SLURM_JOB_ID", "123")

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


def test_run_rejects_invalid_interval(ws):
    result = CliRunner().invoke(main, ["run", "sim_1"], env={MONITOR_ENVVAR: "often"})
    assert result.exit_code != 0
    assert MONITOR_ENVVAR in result.output
    assert not (ws / "sim_1" / "a").exists()
