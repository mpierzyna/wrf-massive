# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from fixtures import simple_simulation

from wrf_massive.base import Resources
from wrf_massive.stages.wrf import wrf as wrf_mod
from wrf_massive.stages.wrf.wrf import WRFStage, is_wrf_successful


def test_is_wrf_successful_requires_wrfout_and_success_marker(tmp_path):
    (tmp_path / "rsl.out.0000").write_text("SUCCESS COMPLETE WRF\n")
    assert not is_wrf_successful(tmp_path)  # success marker but no wrfout
    (tmp_path / "rsl.out.0000").unlink()

    (tmp_path / "wrfout_d01_2020-07-01").write_text("x")
    assert not is_wrf_successful(tmp_path)  # wrfout present but run crashed before completing
    (tmp_path / "rsl.error.0000").write_text("Timing for main\n SUCCESS COMPLETE WRF\n")
    assert is_wrf_successful(tmp_path)


def _make_wrf_stage(sim_dir, resources) -> WRFStage:
    nml = sim_dir / "namelist.tmpl.input"
    nml.write_text("&domains\n max_dom = 1,\n/\n")
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


@pytest.mark.parametrize(
    "resources, n_cpus",
    [
        (Resources(n_tasks=8, cpus_per_task=1, mem_per_cpu="1G"), "8"),
        (None, None),  # inherit the parent environment (SLURM-driven, or serial fallback)
    ],
)
def test_run_passes_n_cpus_env(monkeypatch, simple_simulation, resources, n_cpus):
    captured = {}
    monkeypatch.setattr(wrf_mod, "run_cmd_logged", lambda *a, env=None, **k: captured.update(env=env))
    stage = _make_wrf_stage(simple_simulation.sim_dir, resources)
    stage.run(simple_simulation)
    if n_cpus is None:
        assert captured["env"] is None
    else:
        assert captured["env"]["N_CPUS"] == n_cpus
