from fixtures import simple_simulation

from wrf_massive.base import Resources
from wrf_massive.stages.wrf import wrf as wrf_mod
from wrf_massive.stages.wrf.wrf import WRFStage, is_wrf_successful


def test_is_wrf_successful_requires_wrfout(tmp_path):
    (tmp_path / "rsl.out.0000").write_text("SUCCESS COMPLETE WRF\n")
    assert not is_wrf_successful(tmp_path)  # success marker but no wrfout


def test_is_wrf_successful_requires_success_marker(tmp_path):
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


def test_run_passes_n_cpus_env(monkeypatch, simple_simulation):
    captured = {}
    monkeypatch.setattr(wrf_mod, "run_cmd_logged", lambda *a, env=None, **k: captured.update(env=env))
    stage = _make_wrf_stage(simple_simulation.sim_dir, Resources(n_tasks=8, cpus_per_task=1, mem_per_cpu="1G"))
    stage.run(simple_simulation)
    assert captured["env"] is not None and captured["env"]["N_CPUS"] == "8"


def test_run_without_resources_inherits_env(monkeypatch, simple_simulation):
    captured = {}
    monkeypatch.setattr(wrf_mod, "run_cmd_logged", lambda *a, env=None, **k: captured.update(env=env))
    stage = _make_wrf_stage(simple_simulation.sim_dir, None)
    stage.run(simple_simulation)
    assert captured["env"] is None  # inherit the parent environment (SLURM-driven, or serial fallback)
