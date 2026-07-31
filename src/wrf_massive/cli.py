from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Tuple

import pathlib
import random
from typing import List
import subprocess

import importlib

import click

from wrf_massive.base import Pipeline, Simulation
from wrf_massive.log import get_logger

logger = get_logger()
DEBUG = False  # will be toggled by CLI option


def get_walltime_str(walltime: datetime.timedelta) -> str:
    """Walltime as string in format HH:MM:SS for slurm"""
    hours, remainder = divmod(walltime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_pipeline_cli(p: Pipeline) -> click.Group:
    """Get CLI to control a given pipeline."""

    def _parse_stages(stages: str | None) -> List[str]:
        """Parse stages argument into list of stage names. If None, return all stages."""
        if isinstance(stages, str):
            if "," in stages:
                return stages.split(",")
            return [stages]
        else:
            return p.stage_names

    def _validate_stages(stages: List[str], require_resources: bool) -> List[str]:
        """Validate that all stages exist in pipeline. If required, make sure resources are defined."""
        for stage_name in stages:
            if stage_name not in p.stage_names:
                raise click.UsageError(f"Stage '{stage_name}' not found in pipeline. Available stages: {p.stage_names}")
            if require_resources and p[stage_name].resources is None:
                raise click.UsageError(f"Stage '{stage_name}' does not have resources specified. Cannot submit.")
        return stages

    def _submit_stage_slurm(
        *,
        sim_dir: pathlib.Path,
        stage_name: str,
        jobfile: str,
        dep_job_id: int | None = None,
    ) -> int:
        """Submit a single stage of a simulation to SLURM."""
        # Get simulation and stage
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

        # Submit job
        cmd = [
            # Sbatch command and its arguments
            "sbatch",
            *sbatch_args,
            str(jobfile),
            # Arguments for run command
            stage_name,
            str(sim_dir),
        ]
        if DEBUG:
            logger.debug("Debug mode: not actually submitting job to SLURM.")
            logger.debug(cmd)
            return random.randint(0, 1000)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"Failed to submit job. Return code: {proc.returncode}. {proc.stderr}")
            raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)

        job_id = int(proc.stdout.strip())
        lane_msg = f" (lane {lane}/{max_concurrent})" if lane is not None else ""
        logger.info(f"Submitted job {job_id}{f' depending on {dep_job_id}' if dep_job_id else ''}{lane_msg}.")

        return job_id

    @click.group()
    @click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
    def cli(debug: bool):
        if debug:
            global DEBUG
            DEBUG = True
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled.")

    @cli.command()
    @click.option("-r", "--root", type=click.Path(exists=True, dir_okay=True), default=".")
    @click.argument("sim_module", type=click.Path(exists=True, dir_okay=False))
    @click.argument("sim_name", type=str)
    def init_sims(root: str, sim_module: str, sim_name: str):
        """Initialize simulation dirs and config files from variable in module.

        Parameters
        ----------
        root : str
            Root directory to create simulation dirs in. Default: current dir.
        sim_module : str
            Path to module containing simulations (e.g., `simulations.py`).
        sim_name : str
            Name of variable in module containing simulation or list of simulations.
        """
        # Load module containing simulations
        if sim_module.endswith(".py"):
            sim_module = sim_module.replace(".py", "")
        sim_module = importlib.import_module(sim_module)
        click.echo(f"Loaded module {sim_module}.")

        # Select simulations
        sims = getattr(sim_module, sim_name)
        if not isinstance(sims, list):
            sims = [sims]
        click.echo(f"Found {len(sims)} simulation(s) in variable '{sim_name}'.")

        # Init simulations
        for s in sims:
            s: Simulation
            if s.sim_dir.exists():
                click.echo(f"Simulation dir '{s.sim_dir}' already exists. It will be updated..")
            s.to_disk(root)
            click.echo(f"Initialized simulation <{s}> in '{s.sim_dir}'.")

    @cli.command()
    @click.option("--stages", "-s", type=str, required=False, default=None)
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
    @click.argument("sim_dirs", type=click.Path(exists=True, dir_okay=True), nargs=-1)
    def run(stages: str | None, sim_dirs: Tuple[str, ...], force_setup: bool, force_run: bool):
        """Run stages for simulations.

        Parameters
        ----------
        stages : str | None
            Comma-separated list of stages to run. If None, all stages will be run.
            If multiple stages are specified, they will be run sequentially.
        sim_dirs : Tuple[str, ...]
            Simulation directories to run. At least one must be provided.
        """
        if not sim_dirs:
            raise click.UsageError("At least one simulation directory must be provided.")

        stages = _parse_stages(stages)
        stages = _validate_stages(stages, require_resources=False)
        for sim_dir in sim_dirs:
            s = Simulation.from_disk(sim_dir)
            click.echo(f"Running stages {stages} for simulation <{s}> from '{s.sim_dir}'...")
            if force_setup:
                click.echo("Force setup enabled.")
            if force_run:
                click.echo("Force run enabled.")
            p.run(s, stages=stages, force_setup=force_setup, force_run=force_run)

    @cli.command()
    @click.option("--stages", "-s", type=str, required=False, default=None)
    @click.option(
        "--all",
        "all_files",
        is_flag=True,
        default=False,
        help="Move the ENTIRE work dir back, ignoring each stage's tmp_teardown_globs.",
    )
    @click.argument("sim_dirs", type=click.Path(exists=True, dir_okay=True), nargs=-1)
    def teardown(stages: str | None, all_files: bool, sim_dirs: Tuple[str, ...]):
        """Move stage work dirs back from temporary storage.

        This forces the teardown that normally happens when a stage finishes, overriding
        `tmp_skip_teardown`. Use it to reclaim results left on temporary storage by a stage configured
        with `tmp_skip_teardown`, by a failed run, or by an already-completed stage. Stages that are not
        on temporary storage are skipped.

        Parameters
        ----------
        stages : str | None
            Comma-separated list of stages to tear down. If None, all stages will be torn down.
        all_files : bool
            Move the entire work dir back, ignoring each stage's `tmp_teardown_globs`.
        sim_dirs : Tuple[str, ...]
            Simulation directories to tear down. At least one must be provided.
        """
        if not sim_dirs:
            raise click.UsageError("At least one simulation directory must be provided.")

        stages = _parse_stages(stages)
        stages = _validate_stages(stages, require_resources=False)
        for sim_dir in sim_dirs:
            s = Simulation.from_disk(sim_dir)
            click.echo(f"Tearing down stages {stages} for simulation <{s}> from '{s.sim_dir}'...")
            if all_files:
                click.echo("Moving back all files, ignoring teardown globs.")
            p.teardown(s, stages=stages, all_files=all_files)

    @cli.command()
    @click.option("--stages", "-s", type=str, required=False, default=None)
    @click.option("--jobfile", "-j", type=click.Path(exists=True, dir_okay=False), required=True)
    @click.option("--dep-job", "-d", type=int, required=False, default=None, help="Job ID to depend on.")
    @click.argument("sim_dirs", type=click.Path(exists=True, dir_okay=True), nargs=-1)
    def submit(stages: str | None, jobfile: str, dep_job: int | None, sim_dirs: Tuple[str, ...]):
        """Submit stages to be run by SLURM.

        Parameters
        ----------
        stages : str | None
            Comma-separated list of stages to submit. If None, all stages will be submitted.
        jobfile : str
            Path to job script to use for submission.
        dep_job : int | None
            Job ID to depend for first stage. If None, no dependency will be set. This can be used
            to chain an array submission (e.g., WPS with execution limit) to submission of the
            remaining pipeline (this command).
        sim_dirs : Tuple[str, ...]
            Simulation directories to submit. At least one must be provided.
        """
        if not sim_dirs:
            raise click.UsageError("At least one simulation directory must be provided.")

        stages = _parse_stages(stages)
        stages = _validate_stages(stages, require_resources=True)

        # Submit each stage of each simulation with dependency on previous stage
        for sim_dir in sim_dirs:
            prev_job_id = dep_job
            for stage_name in stages:
                click.echo(f"Submitting stage '{stage_name}' for simulation in '{sim_dir}'...")
                job_id = _submit_stage_slurm(
                    sim_dir=pathlib.Path(sim_dir),
                    stage_name=stage_name,
                    jobfile=jobfile,
                    dep_job_id=prev_job_id,
                )
                prev_job_id = job_id

    return cli
