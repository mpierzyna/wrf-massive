# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

from fixtures import simple_simulation

from wrf_massive.base import Resources
from wrf_massive.stages.wrf import wrf as wrf_mod
from wrf_massive.stages.wrf.utils import estimate_ncpus_from_namelist
from wrf_massive.stages.wrf.wrf import WRFStage, is_wrf_successful


def test_is_wrf_successful_requires_wrfout_and_success_marker(tmp_path):
    (tmp_path / "rsl.out.0000").write_text("SUCCESS COMPLETE WRF\n")
    assert not is_wrf_successful(tmp_path)  # success marker but no wrfout
    (tmp_path / "rsl.out.0000").unlink()

    (tmp_path / "wrfout_d01_2020-07-01").write_text("x")
    assert not is_wrf_successful(tmp_path)  # wrfout present but run crashed before completing
    (tmp_path / "rsl.error.0000").write_text("Timing for main\n SUCCESS COMPLETE WRF\n")
    assert is_wrf_successful(tmp_path)


def _write_namelist(path, max_dom=1, e_we="144", e_sn="192"):
    path.write_text(f"&domains\n max_dom = {max_dom},\n e_we = {e_we},\n e_sn = {e_sn},\n/\n")
    return path


def _make_wrf_stage(sim_dir, resources) -> WRFStage:
    nml = _write_namelist(sim_dir / "namelist.tmpl.input")
    tmpl_dir = sim_dir / "wrf_tmpl"
    tmpl_dir.mkdir()
    return WRFStage(
        work_dir="3_wrf",
        met_em_dir="2_wps",
        wrf_tmpl_dir=tmpl_dir,
        namelist_tmpl_path=nml,
        myoutfields_path=None,
        resources=resources,
    )


def test_run_passes_n_cpus_env(monkeypatch, simple_simulation):
    """N_CPUS comes from the resources; without resources the parent environment is inherited (SLURM or serial)."""
    captured = {}
    monkeypatch.setattr(wrf_mod, "run_cmd_logged", lambda *a, env=None, **k: captured.update(env=env))

    for i, (resources, n_cpus) in enumerate(
        [(Resources(n_tasks=8, cpus_per_task=1, mem_per_cpu="1G"), "8"), (None, None)]
    ):
        simple_simulation.sim_dir = simple_simulation.sim_dir.parent / f"sim_{i}"
        simple_simulation.sim_dir.mkdir()
        _make_wrf_stage(simple_simulation.sim_dir, resources).run(simple_simulation)
        assert (captured["env"] is None) if n_cpus is None else (captured["env"]["N_CPUS"] == n_cpus)


def test_estimate_ncpus_from_namelist(tmp_path):
    cases = [  # max_dom, e_we, e_sn, expected
        (1, "144", "192", (2, 44)),  # 144/100*192/100 = 2.8, 144/25*192/25 = 44.2
        (1, "144, 301", "192, 301", (2, 44)),  # values beyond max_dom are ignored
        (2, "144, 301", "192, 301", (9, 44)),  # min of the largest, max of the smallest domain
        (1, "50", "50", (1, 4)),  # lower limit is at least 1
    ]
    for max_dom, e_we, e_sn, expected in cases:
        nml = _write_namelist(tmp_path / "namelist.input", max_dom=max_dom, e_we=e_we, e_sn=e_sn)
        assert estimate_ncpus_from_namelist(nml) == expected, (max_dom, e_we, e_sn)


def test_n_tasks_checked_against_domain_size(caplog, tmp_path):
    """The MPI task count is compared with the domain size and logged (never raised), also for resources assigned
    after construction (e.g. by StageArray); an unreadable grid size (template placeholders) only warns."""

    def mpi_records():
        records = [r for r in caplog.records if "MPI tasks" in r.message or "could not estimate" in r.message]
        caplog.clear()
        return records

    cases = [
        (1, logging.WARNING, "1 MPI tasks are below the recommended range of 2-44 tasks"),
        (4, logging.INFO, "4 MPI tasks are within the recommended range of 2-44 tasks"),
        (64, logging.WARNING, "64 MPI tasks are above the recommended range of 2-44 tasks"),
    ]
    with caplog.at_level(logging.INFO, logger="wrf_massive"):
        for n_tasks, level, message in cases:
            sim_dir = tmp_path / f"tasks_{n_tasks}"
            sim_dir.mkdir()
            _make_wrf_stage(sim_dir, Resources(n_tasks=n_tasks, cpus_per_task=1, mem_per_cpu="1G"))
            [record] = mpi_records()
            assert record.levelno == level and message in record.message

        (tmp_path / "assigned").mkdir()
        stage = _make_wrf_stage(tmp_path / "assigned", None)
        assert not mpi_records()  # nothing to check without resources
        stage.resources = Resources(n_tasks=64, cpus_per_task=1, mem_per_cpu="1G")
        [record] = mpi_records()
        assert record.levelno == logging.WARNING and "above" in record.message

        nml = _write_namelist(tmp_path / "namelist.tmpl.input", e_we="{{ domains__e_we }}")
        (tmp_path / "wrf_tmpl").mkdir()
        WRFStage(
            work_dir="3_wrf",
            met_em_dir="2_wps",
            wrf_tmpl_dir=tmp_path / "wrf_tmpl",
            namelist_tmpl_path=nml,
            myoutfields_path=None,
            resources=Resources(n_tasks=4, cpus_per_task=1, mem_per_cpu="1G"),
        )
        [record] = mpi_records()
        assert record.levelno == logging.WARNING and "could not estimate" in record.message
