from __future__ import annotations

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
        array: str | None = None,
        dep_job_id: int | None = None,
    ) -> int:
        """Submit a single stage of a simulation to SLURM."""
        # Get simulation and stage
        stage = p[stage_name]

        # Make slurm output dir
        slurm_dir: pathlib.Path = sim_dir / "slurm"
        slurm_dir.mkdir(exist_ok=True, parents=True)

        # Construct sbatch command
        slurm_outfile_pattern = "%A_%a.out" if array is not None else "%A.out"
        sbatch_args = [
            "--parsable",  # return job id
            f"--job-name={sim_dir.name}_{stage_name}",
            f"--output={slurm_dir / f'{stage_name}_{slurm_outfile_pattern}'}",
            f"--ntasks={stage.resources.n_tasks}",
            f"--cpus-per-task={stage.resources.cpus_per_task}",
            f"--mem-per-cpu={stage.resources.mem_per_cpu}",
        ]
        if stage.resources.walltime is not None:
            walltime = get_walltime_str(stage.resources.walltime)
            sbatch_args.append(f"--time={walltime}")

        if array is not None:
            sbatch_args.append(f"--array={array}")

        if dep_job_id is not None:
            sbatch_args.append(f"--dependency=afterok:{dep_job_id}")

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
        logger.info(f"Submitted job {job_id}{f' depending on {dep_job_id}' if dep_job_id else ''}.")

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

    @cli.command()
    @click.option("--stages", "-s", type=str, required=False, default=None)
    @click.option("--jobfile", "-j", type=click.Path(exists=True, dir_okay=False), required=True)
    @click.option("--limit", "-l", type=int, required=False, default=4, help="Limit number of array tasks.")
    @click.argument("sim_dirs", type=click.Path(exists=True, dir_okay=True), nargs=-1)
    def submit_array(stages: str | None, jobfile: str, limit: int, sim_dirs: Tuple[str, ...]):
        """Each stage will be submitted as an array of simulations.
        Next stage will only start when all tasks of previous stage are done.
        Use regular `submit` command and `--dep-job` option to chain other jobs to array.

        Parameters
        ----------
        stages : str | None
            Comma-separated list of stages to submit. If None, all stages will be submitted.
            All simulations will be submitted as one array per stage.
        jobfile : str
            Path to job script to use for submission.
        limit : int
            Limit number of array tasks. Default: 4.
        sim_dirs : Tuple[str, ...]
            Simulation directories to submit. At least one must be provided.
        """
        if not sim_dirs:
            raise click.UsageError("At least one simulation directory must be provided.")

        stages = _parse_stages(stages)
        stages = _validate_stages(stages, require_resources=True)

        # Create file with list of simulations to process by array
        sim_dirs = list(sim_dirs)
        array_file = pathlib.Path(".array_sim_dirs")
        if array_file.exists():
            click.echo(f"Array file '{array_file}' with following content already exists: ")
            click.echo(array_file.read_text())
            click.confirm(f"Overwrite?", abort=True)

        array_file.write_text("\n".join(sim_dirs))

        # Line will be selected in jobscript using sed -n $SLURM_ARRAY_TASK_ID. Requires 1-based indexing.
        array = f"1-{len(sim_dirs)}"
        if limit is not None:
            array += f"%{limit}"

        # Submit each stage of each simulation with dependency on previous stage
        prev_job_id = None
        for stage_name in stages:
            click.echo(f"Submitting stage '{stage_name}' as array for {len(sim_dirs)} simulations...")
            job_id = _submit_stage_slurm(
                sim_dir=pathlib.Path("."),
                stage_name=stage_name,
                jobfile=jobfile,
                array=array,
                dep_job_id=prev_job_id,
            )
            prev_job_id = job_id

    return cli
