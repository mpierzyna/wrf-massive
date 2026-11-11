# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The `wrf-massive` command line interface.

Every command except `init` operates on the workspace containing the env file (`--env-file`, default `env.yaml`,
searched from the current directory upwards). Path arguments are relative to the directory the command is run from.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import pathlib
import random
import subprocess
from typing import List, Tuple

import click

from wrf_massive.base import Pipeline, Simulation
from wrf_massive.log import get_logger
from wrf_massive.workspace import (
    DEFAULT_ENV_FILE,
    ENV_FILE_ENVVAR,
    Workspace,
    WorkspaceError,
    init_workspace,
)

logger = get_logger()


class CliState:
    """Global CLI options. The workspace is only loaded when a command needs it, so `--help` works anywhere."""

    def __init__(self, env_file: str, debug: bool):
        self.env_file = env_file
        self.debug = debug
        self._workspace: Workspace | None = None

    @property
    def workspace(self) -> Workspace:
        """The current workspace, activated (logging configured, cwd changed to its root)."""
        if self._workspace is None:
            try:
                ws = Workspace.discover(self.env_file)
                ws.activate()
            except WorkspaceError as e:
                raise click.ClickException(str(e)) from e
            if self.debug:
                logger.setLevel(logging.DEBUG)
                logger.debug("Debug mode enabled.")
            logger.info(f"Workspace '{ws.root}' (env file '{ws.env_file.name}').")
            self._workspace = ws
        return self._workspace

    @property
    def pipeline(self) -> Pipeline:
        ws = self.workspace
        try:
            return ws.pipeline
        except WorkspaceError as e:
            raise click.ClickException(str(e)) from e


pass_state = click.make_pass_decorator(CliState)


def _abspath(path: str | pathlib.Path) -> pathlib.Path:
    """Absolute path w.r.t. the current dir, without resolving symlinks. Needed before the workspace changes dir."""
    return pathlib.Path(os.path.abspath(path))


def get_walltime_str(walltime: datetime.timedelta) -> str:
    """Walltime as string in format HH:MM:SS for slurm"""
    hours, remainder = divmod(int(walltime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_stages(p: Pipeline, stages: str | None) -> List[str]:
    """Parse comma-separated stages argument into list of stage names. If None, return all stages."""
    if stages is None:
        return p.stage_names
    return stages.split(",")


def _validate_stages(p: Pipeline, stages: List[str], require_resources: bool) -> List[str]:
    """Validate that all stages exist in pipeline. If required, make sure resources are defined."""
    for stage_name in stages:
        if stage_name not in p.stage_names:
            raise click.UsageError(f"Stage '{stage_name}' not found in pipeline. Available stages: {p.stage_names}")
        if require_resources and p[stage_name].resources is None:
            raise click.UsageError(f"Stage '{stage_name}' does not have resources specified. Cannot submit.")
    return stages


def _get_stages(p: Pipeline, stages: str | None, require_resources: bool = False) -> List[str]:
    return _validate_stages(p, _parse_stages(p, stages), require_resources=require_resources)


def _require_sim_dirs(sim_dirs: Tuple[str, ...]) -> List[pathlib.Path]:
    if not sim_dirs:
        raise click.UsageError("At least one simulation directory must be provided.")
    return [_abspath(d) for d in sim_dirs]


def _submit_stage_slurm(
    *,
    p: Pipeline,
    ws: Workspace,
    sim_dir: pathlib.Path,
    stage_name: str,
    jobfile: pathlib.Path,
    dep_job_id: int | None = None,
    dry_run: bool = False,
) -> int:
    """Submit a single stage of a simulation to SLURM."""
    stage = p[stage_name]

    # Make slurm output dir
    slurm_dir: pathlib.Path = sim_dir / "slurm"
    slurm_dir.mkdir(exist_ok=True, parents=True)

    # Determine job name, optionally overridden by a singleton-lane assignment. If
    # `max_concurrent` is set on the stage's resources, jobs are hashed (deterministically,
    # across separate CLI invocations) into one of `max_concurrent` lanes sharing an identical
    # job name, and submitted with `--dependency=singleton` so SLURM never runs more than one
    # job per lane at a time -- capping total concurrency for this stage independent of CPU
    # headroom.
    job_name = f"{sim_dir.name}_{stage_name}"
    max_concurrent = stage.resources.max_concurrent
    lane: int | None = None
    if max_concurrent is not None:
        lane = int(hashlib.md5(str(sim_dir).encode()).hexdigest(), 16) % max_concurrent
        job_name = f"{stage_name}_lane{lane}"

    # Construct sbatch command
    sbatch_args = [
        "--parsable",  # return job id
        f"--job-name={job_name}",
        f"--output={slurm_dir / f'{stage_name}_%A.out'}",
        f"--ntasks={stage.resources.n_tasks}",
        f"--cpus-per-task={stage.resources.cpus_per_task}",
        f"--mem-per-cpu={stage.resources.mem_per_cpu}",
    ]
    if stage.resources.walltime is not None:
        walltime = get_walltime_str(stage.resources.walltime)
        sbatch_args.append(f"--time={walltime}")

    # Combine afterok (stage-to-stage) and singleton (lane cap) dependencies; SLURM ANDs
    # comma-separated dependency conditions.
    dep_conditions = []
    if dep_job_id is not None:
        dep_conditions.append(f"afterok:{dep_job_id}")
    if lane is not None:
        dep_conditions.append("singleton")
    if dep_conditions:
        sbatch_args.append(f"--dependency={','.join(dep_conditions)}")

    # Submit job. Arguments after the jobfile are passed to it: stage name and sim dir for `wrf-massive run`.
    cmd = ["sbatch", *sbatch_args, str(jobfile), stage_name, str(sim_dir)]

    # sbatch exports the submitting environment to the job by default, so `wrf-massive run` inside the job
    # uses the same env file.
    env = {**os.environ, ENV_FILE_ENVVAR: str(ws.env_file)}

    if dry_run:
        click.echo(f"Dry run, not submitting: {ENV_FILE_ENVVAR}={ws.env_file} {' '.join(cmd)}")
        return random.randint(0, 1000)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ws.root)
    if proc.returncode != 0:
        logger.error(f"Failed to submit job. Return code: {proc.returncode}. {proc.stderr}")
        raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)

    job_id = int(proc.stdout.strip())
    lane_msg = f" (lane {lane}/{max_concurrent})" if lane is not None else ""
    logger.info(f"Submitted job {job_id}{f' depending on {dep_job_id}' if dep_job_id else ''}{lane_msg}.")

    return job_id


@click.group()
@click.option(
    "--env-file",
    "-e",
    default=DEFAULT_ENV_FILE,
    show_default=True,
    envvar=ENV_FILE_ENVVAR,
    help="Env file of the workspace. Searched in the current directory and its parents; "
    "the directory containing it is the workspace root.",
)
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging; `submit` only prints the jobs.")
@click.pass_context
def main(ctx: click.Context, env_file: str, debug: bool):
    """Orchestrate WRF simulations of a workspace."""
    ctx.obj = CliState(env_file=env_file, debug=debug)


@main.command()
@click.argument("directory", type=click.Path(file_okay=False), default=".")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing files.")
def init(directory: str, force: bool):
    """Create a new workspace in DIRECTORY (default: current dir) from the packaged template.

    The workspace contains an env file, `pipeline.py`, `simulations.py`, WPS/WRF namelist templates and a SLURM job
    script, which serve as starting point for your own setup.
    """
    try:
        written = init_workspace(directory, force=force)
    except WorkspaceError as e:
        raise click.ClickException(str(e)) from e

    root = _abspath(directory)
    for path in written:
        click.echo(f"Created {path.relative_to(root)}")
    click.echo(
        f"\nInitialised workspace in '{root}'. Next steps:\n"
        f"  1. Set the WPS/WRF paths in {DEFAULT_ENV_FILE}.\n"
        f"  2. Set up CDS credentials (~/.cdsapirc) for the forcing download.\n"
        f"  3. Adapt the domain in namelist.tmpl.wps and the simulations in simulations.py.\n"
        f"  4. Run `wrf-massive init-sims simulations.py sim_test`, then `wrf-massive run test_1`."
    )


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory to create simulation dirs in. Default: workspace root.",
)
@click.argument("sim_module", type=click.Path(exists=True, dir_okay=False))
@click.argument("sim_name", type=str)
@pass_state
def init_sims(state: CliState, root: str | None, sim_module: str, sim_name: str):
    """Initialize simulation dirs and config files from variable SIM_NAME in SIM_MODULE.

    SIM_MODULE is the path to a module containing simulations (e.g., `simulations.py`). SIM_NAME is the name of the
    variable in that module holding a simulation or list of simulations.
    """
    sim_module_path = _abspath(sim_module)
    root_path = _abspath(root) if root is not None else None

    ws = state.workspace
    root_path = root_path or ws.root
    module = ws.import_module(sim_module_path)
    click.echo(f"Loaded module {module.__name__} from '{sim_module_path}'.")

    # Select simulations
    try:
        sims = getattr(module, sim_name)
    except AttributeError:
        raise click.UsageError(f"Variable '{sim_name}' not found in '{sim_module_path}'.") from None
    if not isinstance(sims, list):
        sims = [sims]
    click.echo(f"Found {len(sims)} simulation(s) in variable '{sim_name}'.")

    # Init simulations
    for s in sims:
        s: Simulation
        if (root_path / s.sim_dir).exists():
            click.echo(f"Simulation dir '{s.sim_dir}' already exists. It will be updated..")
        s.to_disk(root_path)
        click.echo(f"Initialized simulation <{s}> in '{root_path / s.sim_dir}'.")


@main.command()
@click.option("--stages", "-s", type=str, required=False, default=None, help="Comma-separated stages. Default: all.")
@click.option(
    "--force-setup",
    is_flag=True,
    default=False,
    help="Force rerunning setup of specified stages.",
    envvar="FORCE_SETUP",
)
@click.option(
    "--force-run",
    is_flag=True,
    default=False,
    help="Force rerunning of specified stages.",
    envvar="FORCE_RUN",
)
@click.argument("sim_dirs", type=click.Path(exists=True, file_okay=False), nargs=-1)
@pass_state
def run(state: CliState, stages: str | None, sim_dirs: Tuple[str, ...], force_setup: bool, force_run: bool):
    """Run stages for simulations in SIM_DIRS.

    If multiple stages are specified, they are run sequentially.
    """
    sim_dirs = _require_sim_dirs(sim_dirs)
    p = state.pipeline
    stages = _get_stages(p, stages)
    for sim_dir in sim_dirs:
        s = Simulation.from_disk(sim_dir)
        click.echo(f"Running stages {stages} for simulation <{s}> from '{s.sim_dir}'...")
        if force_setup:
            click.echo("Force setup enabled.")
        if force_run:
            click.echo("Force run enabled.")
        p.run(s, stages=stages, force_setup=force_setup, force_run=force_run)


@main.command()
@click.option("--stages", "-s", type=str, required=False, default=None, help="Comma-separated stages. Default: all.")
@click.option(
    "--all",
    "all_files",
    is_flag=True,
    default=False,
    help="Move the ENTIRE work dir back, ignoring each stage's tmp_teardown_globs.",
)
@click.argument("sim_dirs", type=click.Path(exists=True, file_okay=False), nargs=-1)
@pass_state
def teardown(state: CliState, stages: str | None, all_files: bool, sim_dirs: Tuple[str, ...]):
    """Move stage work dirs of SIM_DIRS back from temporary storage.

    This forces the teardown that normally happens when a stage finishes, overriding `tmp_skip_teardown`. Use it to
    reclaim results left on temporary storage by a stage configured with `tmp_skip_teardown`, by a failed run, or by
    an already-completed stage. Stages that are not on temporary storage are skipped.
    """
    sim_dirs = _require_sim_dirs(sim_dirs)
    p = state.pipeline
    stages = _get_stages(p, stages)
    for sim_dir in sim_dirs:
        s = Simulation.from_disk(sim_dir)
        click.echo(f"Tearing down stages {stages} for simulation <{s}> from '{s.sim_dir}'...")
        if all_files:
            click.echo("Moving back all files, ignoring teardown globs.")
        p.teardown(s, stages=stages, all_files=all_files)


@main.command()
@click.option("--stages", "-s", type=str, required=False, default=None, help="Comma-separated stages. Default: all.")
@click.option("--jobfile", "-j", type=click.Path(exists=True, dir_okay=False), required=True, help="SLURM job script.")
@click.option("--dep-job", "-d", type=int, required=False, default=None, help="Job ID the first stage depends on.")
@click.argument("sim_dirs", type=click.Path(exists=True, file_okay=False), nargs=-1)
@pass_state
def submit(state: CliState, stages: str | None, jobfile: str, dep_job: int | None, sim_dirs: Tuple[str, ...]):
    """Submit stages of simulations in SIM_DIRS to SLURM, one job per stage.

    Stages of a simulation depend on each other; `--dep-job` additionally makes the first stage of every simulation
    wait for another job. The job script is called with the stage name and simulation dir as arguments and should run
    `wrf-massive run --stages=$1 $2`.
    """
    sim_dirs = _require_sim_dirs(sim_dirs)
    jobfile = _abspath(jobfile)
    p = state.pipeline
    stages = _get_stages(p, stages, require_resources=True)

    # Submit each stage of each simulation with dependency on previous stage
    for sim_dir in sim_dirs:
        prev_job_id = dep_job
        for stage_name in stages:
            click.echo(f"Submitting stage '{stage_name}' for simulation in '{sim_dir}'...")
            prev_job_id = _submit_stage_slurm(
                p=p,
                ws=state.workspace,
                sim_dir=sim_dir,
                stage_name=stage_name,
                jobfile=jobfile,
                dep_job_id=prev_job_id,
                dry_run=state.debug,
            )


@main.group()
def plot():
    """Plot WPS domains. Works without a workspace."""


@plot.command()
@click.argument("geo_em_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--cmap", default="terrain", show_default=True, help="Matplotlib colormap for terrain height.")
@click.option("--coastlines/--no-coastlines", default=True, show_default=True, help="Draw coastlines.")
@click.option("--coastline-resolution", type=click.Choice(["10m", "50m", "110m"]), default="10m", show_default=True)
@click.option("--gridlines/--no-gridlines", default=True, show_default=True, help="Draw labelled lat/lon gridlines.")
def domain_geo_em(geo_em_path: str, cmap: str, coastlines: bool, coastline_resolution: str, gridlines: bool):
    """Plot the terrain of a domain from GEO_EM_PATH (a geo_em.dNN.nc file written by geogrid).

    The plot is saved next to the file as `<name>_domains.png`.
    """
    from wrf_massive.plotting import plot_domain_geo_em  # imported lazily: plotting pulls in xarray/matplotlib

    geo_em_path = pathlib.Path(geo_em_path)
    output = geo_em_path.with_name(geo_em_path.stem + "_domains.png")
    plot_domain_geo_em(
        geo_em_path,
        cmap=cmap,
        add_coastlines=coastlines,
        coastline_resolution=coastline_resolution,
        add_gridlines=gridlines,
        savefig=output,
    )
    click.echo(f"-> Saved domain plot to {output}")


@plot.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,  # `--help` is handled below, to include the options of plot_domains.py
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def domain_nml(args: Tuple[str, ...]):
    """Plot the domain layout of a WPS namelist without running geogrid (`--help` for options).

    Usage: `wrf-massive plot domain-nml NAMELIST [OPTIONS]`, e.g. `wrf-massive plot domain-nml namelist.tmpl.wps`.
    Runs `plot_domains.py` from WPS v4.7.0 (Anthony Islas, 2025) on NAMELIST, saving to `domains.png` unless
    `--save_file` is given. OPTIONS are passed on to the script.
    """
    from wrf_massive.plotting import plot_domain_nml  # imported lazily: plotting pulls in xarray/matplotlib

    if not args or args[0] in ("-h", "--help"):
        click.echo(
            "Usage: wrf-massive plot domain-nml NAMELIST [OPTIONS]\n\n"
            "Plot the domain layout of NAMELIST (a namelist.wps or namelist.tmpl.wps) without running geogrid,\n"
            "using plot_domains.py of WPS v4.7.0 (Anthony Islas, 2025). The plot is saved to domains.png unless\n"
            "--save_file is given. OPTIONS of plot_domains.py (-n/--namelist is set from NAMELIST):\n"
        )
        if not args:
            raise click.UsageError("Missing argument 'NAMELIST'.")
        plot_domain_nml(None, ["--help"])
        return

    namelist, *options = args
    if not pathlib.Path(namelist).is_file():
        raise click.UsageError(f"NAMELIST '{namelist}' is not a file. It must be the first argument.")
    plot_domain_nml(namelist, options)


if __name__ == "__main__":
    main()
