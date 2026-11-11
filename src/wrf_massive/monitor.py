# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optional monitoring of memory and disk usage while a stage runs (`monitor.sh`, started by `wrf-massive run`)."""

from __future__ import annotations

import contextlib
import datetime
import importlib.resources
import os
import pathlib
import subprocess
import time
from typing import Iterator

from wrf_massive.log import get_logger

logger = get_logger()

MONITOR_ENVVAR = "WRF_MASSIVE_MONITOR"  # sampling interval in seconds; unset or empty disables monitoring
STARTUP_TIMEOUT_S = 10


def get_monitor_interval() -> int | None:
    """Sampling interval from $WRF_MASSIVE_MONITOR, or None if monitoring is disabled."""
    value = os.environ.get(MONITOR_ENVVAR, "").strip()
    if not value:
        return None
    try:
        interval = int(value)
    except ValueError:
        interval = 0
    if interval <= 0:
        raise ValueError(f"${MONITOR_ENVVAR} must be a positive number of seconds, got '{value}'.")
    return interval


def monitor_prefix(sim_dir: pathlib.Path, stage_name: str) -> pathlib.Path:
    """`<sim_dir>/slurm/<stage>_<SLURM job id>`, or `..._local-<timestamp>` outside SLURM, next to the job output."""
    job_id = os.environ.get("SLURM_JOB_ID") or f"local-{datetime.datetime.now():%Y%m%dT%H%M%S}"
    return sim_dir / "slurm" / f"{stage_name}_{job_id}"


@contextlib.contextmanager
def monitor(interval: int | None, work_dir: pathlib.Path, out_prefix: pathlib.Path) -> Iterator[None]:
    """Log memory and disk usage of `work_dir` every `interval` seconds to `<out_prefix>_{mem,disk}.csv`.

    Does nothing if `interval` is None. The monitor runs as a child process, so in a SLURM job it is part of the
    job's cgroup, whose memory it reports. It takes a last sample when stopped.
    """
    if interval is None:
        yield
        return

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mem_csv = out_prefix.with_name(out_prefix.name + "_mem.csv")
    script = importlib.resources.files("wrf_massive").joinpath("monitor.sh")
    # A background process, so not `run_cmd_logged` (which blocks until the command is done)
    proc = subprocess.Popen(["bash", str(script), str(interval), str(work_dir), str(out_prefix)])

    # Wait for the first sample: the script then has its stop handler in place and writes a last sample when stopped
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while proc.poll() is None and time.monotonic() < deadline:
        if mem_csv.exists() and len(mem_csv.read_text().splitlines()) > 1:
            break
        time.sleep(0.05)
    logger.info(f"Monitoring memory and disk usage every {interval} s to {out_prefix}_{{mem,disk}}.csv")

    try:
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
