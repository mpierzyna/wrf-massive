# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The `wrf-massive` command line interface.

Every command except `init` and `plot` operates on the workspace containing `env.yaml`, searched from the current
directory upwards. Path arguments are relative to the directory the command is run from.
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
from wrf_massive.monitor import get_monitor_interval, monitor, monitor_prefix
from wrf_massive.workspace import (
    ENV_FILE,
    ENV_HASH_ENVVAR,
    INIT_ENV_FILE,
    Workspace,
    WorkspaceError,
    import_module,
    init_workspace,
)

logger = get_logger()


class CliState:
    """Global CLI options. The workspace is only loaded when a command needs it, so `--help` works anywhere."""

    def __init__(self, debug: bool):
        self.debug = debug
        self._workspace: Workspace | None = None

    @property
    def workspace(self) -> Workspace:
        """The current workspace, activated (logging configured, cwd changed to its root).

        In a job started by `submit`, fails if env.yaml changed since the submission.
        """
        if self._workspace is None:
            try:
                ws = Workspace.discover()
                ws.check_env_unchanged(os.environ.get(ENV_HASH_ENVVAR))
                ws.activate()
            except WorkspaceError as e:
                raise click.ClickException(str(e)) from e
            if self.debug:
                logger.setLevel(logging.DEBUG)
                logger.debug("Debug mode enabled.")
            target = f" -> {os.readlink(ws.env_file)}" if ws.env_file.is_symlink() else ""
            logger.info(f"Workspace '{ws.root}' ({ENV_FILE}{target}).")
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


def _load_sims_from_module(sim_module_path: pathlib.Path, sim_names: Tuple[str, ...]) -> List[Simulation]:
    """Simulations of the variables `sim_names` in module `sim_module_path`, each holding a simulation or a list."""
    module = import_module(sim_module_path)
    click.echo(f"Loaded module {module.__name__} from '{sim_module_path}'.")
    sims = []
    for sim_name in sim_names:
        try:
            value = getattr(module, sim_name)
        except AttributeError:
            raise click.UsageError(f"Variable '{sim_name}' not found in '{sim_module_path}'.") from None
        if not isinstance(value, list):
            value = [value]
        if not all(isinstance(s, Simulation) for s in value):
            raise click.UsageError(f"Variable '{sim_name}' must hold a Simulation or a list of Simulations.")
        click.echo(f"Found {len(value)} simulation(s) in variable '{sim_name}'.")
        sims.extend(value)
    return sims


def _submit_stage_slurm(
    *,
    p: Pipeline,
    ws: Workspace,
    sim_dir: pathlib.Path,
    stage_name: str,
    jobfile: pathlib.Path,
    job_env: dict[str, str],
    dep_job_id: int | None = None,
    dry_run: bool = False,
) -> int:
    """Submit a single stage of a simulation to SLURM, passing `job_env` to the job via `--export=ALL,...`."""
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
        f"--nodes={stage.resources.n_nodes}",
        f"--ntasks={stage.resources.n_tasks}",
        f"--cpus-per-task={stage.resources.cpus_per_task}",
        f"--mem-per-cpu={stage.resources.mem_per_cpu}",
    ]
    if stage.resources.walltime is not None:
        walltime = get_walltime_str(stage.resources.walltime)
        sbatch_args.append(f"--time={walltime}")
    if stage.resources.partition is not None:
        sbatch_args.append(f"--partition={stage.resources.partition}")  # command line beats `#SBATCH` in job file

    # Combine afterok (stage-to-stage) and singleton (lane cap) dependencies; SLURM ANDs
    # comma-separated dependency conditions.
    dep_conditions = []
    if dep_job_id is not None:
        dep_conditions.append(f"afterok:{dep_job_id}")
    if lane is not None:
        dep_conditions.append("singleton")
    if dep_conditions:
        sbatch_args.append(f"--dependency={','.join(dep_conditions)}")

    # Pass `job_env` explicitly with `--export`: some clusters (e.g. Snellius) drop variables that are only set in the
    # environment of the sbatch process, even with the default `--export=ALL`. sbatch splits the list on commas, so
    # values must not contain any.
    if job_env:
        for k, v in job_env.items():
            if "," in k or "," in v:
                raise ValueError(
                    f"Job environment variable {k}={v!r} contains a comma, which sbatch --export cannot pass."
                )
        sbatch_args.append("--export=" + ",".join(["ALL", *(f"{k}={v}" for k, v in job_env.items())]))

    # Submit job. Arguments after the jobfile are passed to it: stage name and sim dir for `wrf-massive run`.
    cmd = ["sbatch", *sbatch_args, str(jobfile), stage_name, str(sim_dir)]

    if dry_run:
        click.echo(f"Dry run, not submitting: {' '.join(cmd)}")
        return random.randint(0, 1000)

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ws.root)
    if proc.returncode != 0:
        logger.error(f"Failed to submit job. Return code: {proc.returncode}. {proc.stderr}")
        raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)

    job_id = int(proc.stdout.strip())
    lane_msg = f" (lane {lane}/{max_concurrent})" if lane is not None else ""
    logger.info(f"Submitted job {job_id}{f' depending on {dep_job_id}' if dep_job_id else ''}{lane_msg}.")

    return job_id


@click.group()
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging; `submit` only prints the jobs.")
@click.pass_context
def main(ctx: click.Context, debug: bool):
    """Orchestrate WRF simulations of a workspace.

    The workspace is the closest directory (current dir or a parent) containing `env.yaml`. To switch between
    machine-specific settings, symlink the env file of the machine: `ln -sf env_<machine>.yaml env.yaml`.
    """
    ctx.obj = CliState(debug=debug)


@main.command()
@click.argument("directory", type=click.Path(file_okay=False), default=".")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing files.")
def init(directory: str, force: bool):
    """Create a new workspace in DIRECTORY (default: current dir) from the packaged template.

    The workspace contains an env file (`env_local.yaml`, symlinked to `env.yaml`), `pipeline.py`, `simulations.py`, WPS/WRF namelist templates and a SLURM job
    script, which serve as starting point for your own setup.
    """
    try:
        written = init_workspace(directory, force=force)
    except WorkspaceError as e:
        raise click.ClickException(str(e)) from e

    root = _abspath(directory)
    for path in written:
        target = f" -> {os.readlink(path)}" if path.is_symlink() else ""
        click.echo(f"Created {path.relative_to(root)}{target}")
    click.echo(
        f"\nInitialised workspace in '{root}'. Next steps:\n"
        f"  1. Set the WPS/WRF paths in {INIT_ENV_FILE} ({ENV_FILE} links to it; rename it per machine if you like).\n"
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
    sims = _load_sims_from_module(sim_module_path, (sim_name,))

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

    If multiple stages are specified, they are run sequentially. If $WRF_MASSIVE_MONITOR is set to a number of
    seconds, the memory and disk usage of each stage is logged at that interval to
    `<sim_dir>/slurm/<stage>_<job id>_{mem,disk}.csv`.
    """
    sim_dirs = _require_sim_dirs(sim_dirs)
    try:
        interval = get_monitor_interval()
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    p = state.pipeline
    stages = _get_stages(p, stages)
    for sim_dir in sim_dirs:
        s = Simulation.from_disk(sim_dir)
        click.echo(f"Running stages {stages} for simulation <{s}> from '{s.sim_dir}'...")
        if force_setup:
            click.echo("Force setup enabled.")
        if force_run:
            click.echo("Force run enabled.")
        if interval is None:
            p.run(s, stages=stages, force_setup=force_setup, force_run=force_run)
            continue
        for stage_name in stages:  # one monitor per stage, on that stage's work dir
            work_dir = p[stage_name].get_work_dir(s, create=False)
            with monitor(interval, work_dir, monitor_prefix(s.sim_dir, stage_name)):
                p.run(s, stages=[stage_name], force_setup=force_setup, force_run=force_run)


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
@click.option(
    "--jobfile",
    "-j",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="SLURM job script. Overrides `job_file` of the env file.",
)
@click.option("--dep-job", "-d", type=int, required=False, default=None, help="Job ID the first stage depends on.")
@click.option(
    "--force-setup",
    is_flag=True,
    default=False,
    help="Make the jobs rerun the setup of the submitted stages.",
    envvar="FORCE_SETUP",
)
@click.argument("sim_dirs", type=click.Path(exists=True, file_okay=False), nargs=-1)
@pass_state
def submit(
    state: CliState,
    stages: str | None,
    jobfile: str | None,
    dep_job: int | None,
    force_setup: bool,
    sim_dirs: Tuple[str, ...],
):
    """Submit stages of simulations in SIM_DIRS to SLURM, one job per stage.

    Stages of a simulation depend on each other; `--dep-job` additionally makes the first stage of every simulation
    wait for another job. The job script is `job_file` of the env file, unless `--jobfile` is given. It is called with
    the stage name and simulation dir as arguments and should run `wrf-massive run --stages=$1 $2`. With
    `--force-setup`, the jobs get `FORCE_SETUP=1` in their environment, so that `wrf-massive run` reruns the setup of
    the stage like `run --force-setup` (a `.done` marker still skips the stage).
    """
    sim_dirs = _require_sim_dirs(sim_dirs)
    jobfile = _abspath(jobfile) if jobfile is not None else None
    p = state.pipeline
    if jobfile is None:
        ws = state.workspace
        jobfile = ws.job_file
        if jobfile is None:
            raise click.UsageError(f"No job script: set `job_file` in '{ws.env_file}' or pass --jobfile.")
        if not jobfile.is_file():
            raise click.UsageError(f"Job script '{jobfile}' (`job_file` in '{ws.env_file}') does not exist.")
    stages = _get_stages(p, stages, require_resources=True)

    # Passed to `wrf-massive run` in the jobs through their environment, so they work with any job script: the env
    # hash lets the job check that env.yaml is still the one it was submitted with, FORCE_SETUP is `run --force-setup`.
    job_env = {ENV_HASH_ENVVAR: state.workspace.env_hash}
    if force_setup:
        job_env["FORCE_SETUP"] = "1"

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
                job_env=job_env,
                dep_job_id=prev_job_id,
                dry_run=state.debug,
            )


@main.group("store")
def store_group():
    """Zarr store of post-processed output of many domains (needs the `store` extra)."""


@store_group.command("init-domain")
@click.option("--stage", "stage_name", required=True, help="Stage in pipeline.py that writes to a ZarrStore.")
@click.argument("sim_dirs", type=click.Path(exists=True, file_okay=False), nargs=-1)
@pass_state
def store_init_domain(state: CliState, stage_name: str, sim_dirs: Tuple[str, ...]):
    """Add the domain of the given postproc stage and the given sims to the store.

    Creates a new store if this is the first domain, appends the domain to an existing store otherwise, and appends
    sims not yet in the domain. Creating a domain takes its layout from one processed frame of the first given sim
    with output. Only metadata is written; no space is reserved. With `wrf-massive --debug`, only prints what would
    be done.
    """
    sim_dirs = _require_sim_dirs(sim_dirs)
    p = state.pipeline
    if stage_name not in p.stages:
        raise click.UsageError(f"Unknown stage '{stage_name}'. Stages of the pipeline: {list(p.stages)}.")
    from wrf_massive.store.zarr_store.init import init_domain  # needs zarr: only here

    sims = [Simulation.from_disk(d) for d in sim_dirs]
    try:
        report = init_domain(p[stage_name], sims, dry_run=state.debug)
    except (ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    for line in report:
        click.echo(line)


@store_group.command("verify")
@click.option("--stage", "stage_name", default=None, help="Stage in pipeline.py whose ZarrStore to check.")
@click.option(
    "--path",
    "store_path",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Store (or a copy) to check instead; no workspace needed.",
)
@click.option("--domain", "domains", multiple=True, help="Only this domain (repeatable).")
@click.option(
    "--clean", is_flag=True, default=False, help="Remove stray `*.partial` files (only while nothing writes)."
)
@click.argument("sim_dirs", type=click.Path(exists=True, file_okay=False), nargs=-1)
@pass_state
def store_verify(state: CliState, stage_name, store_path, domains, clean, sim_dirs):
    """Report what a store or a local copy contains: per domain sims, frames written, variables and size.

    With --stage: the stage's store and domain, limited to the sims in SIM_DIRS if given. With --path: any store or
    copy (made by `store pull`), no workspace needed.
    """
    if (stage_name is None) == (store_path is None):
        raise click.UsageError("Give either --stage or --path.")
    from wrf_massive.store.zarr_store.verify import verify  # needs zarr: only here

    sims = None
    if stage_name is not None:
        p = state.pipeline
        if stage_name not in p.stages:
            raise click.UsageError(f"Unknown stage '{stage_name}'. Stages of the pipeline: {list(p.stages)}.")
        from wrf_massive.store.zarr_store.store import zarr_store_of

        try:
            cfg = zarr_store_of(p[stage_name])
        except ValueError as e:
            raise click.UsageError(str(e)) from e
        store_path, domains = cfg.abs_path, domains or (cfg.domain,)
        sims = [Simulation.from_disk(d).name for d in _require_sim_dirs(sim_dirs)] if sim_dirs else None
    elif sim_dirs:
        raise click.UsageError("SIM_DIRS need --stage.")
    try:
        report = verify(_abspath(store_path), domains=list(domains) or None, sims=sims, clean=clean)
    except (FileNotFoundError, KeyError) as e:
        raise click.ClickException(str(e)) from e
    for line in report:
        click.echo(line)


@store_group.command("pull")
@click.argument("src")
@click.argument("dst", type=click.Path(file_okay=False))
@click.option("--domain", "domains", multiple=True, required=True, help="Domain to copy (repeatable).")
@click.option("--vars", "variables", default=None, help="Only these variables, comma-separated (default: all).")
@click.option("--sims", default=None, help="Only these sims by name, comma-separated, globs allowed (default: all).")
@click.option(
    "--rclone-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="rclone config file (passed as `rclone --config`; default: rclone's own).",
)
@click.option("--transfers", type=int, default=8, show_default=True, help="Parallel transfers.")
@click.option("--dry-run", is_flag=True, default=False, help="Copy only metadata and show what would be copied.")
def store_pull(src, dst, domains, variables, sims, rclone_config, transfers, dry_run):
    """Copy (part of) a store with rclone. SRC is an rclone path (`remote:path` or local), DST a local dir.

    Metadata, coordinates, static fields and the `written` mask are always copied; data arrays are filtered by
    --vars and --sims. A manifest in DST records what the copy holds, so readers raise instead of returning NaN
    for anything else. Pulling again into the same DST adds to the copy. Works without a workspace.
    """
    from wrf_massive.store.zarr_store.pull import pull  # needs zarr: only here

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None  # noqa: E731
    try:
        report = pull(
            src,
            _abspath(dst),
            list(domains),
            variables=split(variables),
            sims=split(sims),
            rclone_config=_abspath(rclone_config) if rclone_config else None,
            transfers=transfers,
            dry_run=dry_run,
        )
    except (KeyError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    for line in report:
        click.echo(line)


@main.group()
def plot():
    """Plot WPS domains and simulation timelines. Works without a workspace."""


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


@plot.command()
@click.option(
    "-m",
    "--module",
    "sim_module",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Read TARGETS as variable names in this module (like `init-sims`) instead of simulation dirs.",
)
@click.option("--start", type=click.DateTime(), default=None, help="Timeline start. Default: earliest sim begin.")
@click.option("--end", type=click.DateTime(), default=None, help="Timeline end. Default: latest sim end.")
@click.argument("targets", nargs=-1)
def sim_timeline(
    sim_module: str | None,
    start: datetime.datetime | None,
    end: datetime.datetime | None,
    targets: Tuple[str, ...],
):
    """Plot the periods covered by simulations (warm-up excluded) on a timeline.

    TARGETS are simulation dirs (as created by `init-sims`). With `-m SIM_MODULE`, TARGETS are instead names of
    variables in SIM_MODULE, each holding a simulation or list of simulations. The title names the module and
    variables, or the alphabetically first and last simulation dir, plus the number of simulations and their total
    usable days. The plot is saved in the current dir as `timeline_<module stem>_<variables>.png` or
    `timeline_<first dir>...<last dir>.png`.
    """
    from wrf_massive.plotting import plot_sim_timeline  # imported lazily: plotting pulls in xarray/matplotlib

    if not targets:
        what = "variable name" if sim_module is not None else "simulation directory"
        raise click.UsageError(f"At least one {what} must be provided.")

    if sim_module is not None:
        sim_module_path = _abspath(sim_module)
        sims = _load_sims_from_module(sim_module_path, targets)
        label = f"{sim_module_path.name}: {', '.join(targets)}"
        out_name = f"timeline_{sim_module_path.stem}_{'_'.join(targets)}.png"
    else:
        sims = []
        for sim_dir in targets:
            if not (pathlib.Path(sim_dir) / "simulation.yaml").is_file():
                raise click.UsageError(f"'{sim_dir}' is not a simulation dir (no simulation.yaml).")
            sims.append(Simulation.from_disk(sim_dir))
        click.echo(f"Loaded {len(sims)} simulation(s).")
        names = sorted(_abspath(t).name for t in targets)
        label = names[0] if len(names) == 1 else f"{names[0]}...{names[-1]}"
        out_name = f"timeline_{label.replace(' ', '_')}.png"

    days_total = sum((s.end - s.begin for s in sims), datetime.timedelta()) / datetime.timedelta(days=1)
    try:
        plot_sim_timeline(
            sims,
            start=start,
            end=end,
            title=f"{label} (n={len(sims)} sims, {days_total:g} days)",
            savefig=out_name,
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    click.echo(f"-> Saved simulation timeline to {out_name}")


if __name__ == "__main__":
    main()
