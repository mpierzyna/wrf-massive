# WRF Massive — example workspace

This README explains how to set up and run an example WRF study contained in workspaces/example.
It describes what the `pipeline.py` and `simulations.py` files do, what the `env_*.yaml` files are for, and how to use the provided CLI to initialize, run, or submit simulations.

> [!NOTE]
> This example does not run by itself. It is intended as a starting point to 
> to set up your own project.
>
> For a fully functional setup, visit the [OTProf](https://github.com/mpierzyna/otprof/tree/main/data/nl/wrf) project, where I used
> wrf-massive to generate 1-year of CERRA-forced WRF simulations of the Netherlands.


## Overview

This example workspace contains a small WRF pipeline and a test simulation. The pipeline orchestrates staged work for each simulation (fetch forcing, run WPS, run WRF, postprocess). The provided CLI wraps common tasks: creating per-simulation directories, running stages locally, and submitting stages to SLURM.

Important files in this directory
- `pipeline.py` — Defines the pipeline(s) used for runs. The pipeline is assembled from stage objects and `Resources` declarations.
- `simulations.py` — Contains one or more `Simulation` objects (configuration + times) that are used to create per-simulation directories and configuration files.
- `env_*.yaml` (`env_dev.yaml`, `env_hpc.yaml`) — Host-specific settings (paths to WRF/WPS templates, which pipeline to use, machine name). Copy one of these to `env.yaml` before running.
- `cli.py` — Convenience script that loads `env.yaml`, selects the pipeline, and invokes the pipeline CLI (from the `wrf_massive` package).
- `namelist.tmpl.input`, `namelist.tmpl.wps` — Jinja-like templates used to render WRF/WPS namelists using `Simulation` settings.
- `myoutfields.txt` — Example additional WRF output fields.
- `slurm_hpc.sh` — Example SLURM job script for this workspace.

## How the pipeline works (high-level)

`pipeline.py` builds a `Pipeline` instance composed of named stages. Each stage is a `Stage`-derived object (for example, `PullCerraStage`, `WPSStage`, `WRFStage`, `PostprocCn2Stage`) and has an associated `Resources` object describing how many tasks, CPUs and memory should be used when submitting that stage to a scheduler.

The example defines at least the following pipelines (names can differ by environment):
- `p_default` — A straightforward pipeline with `cerra` (forcing), `wps` and `wrf` stages plus a `sim_done` marker (used for local/development runs).
- `p_hpc` — An HPC-tailored variant (defined when `env["machine"] == "hpc"` in `pipeline.py`) that assumes WPS/forcing may be handled externally and increases WRF resource allocations.

Stages are executed in-order by name when the pipeline is asked to `run(...)`. On clusters, the provided CLI can submit each stage as a SLURM job or an array of jobs.

## Defining simulations (`simulations.py`)

Simulations are represented by `Simulation` objects (from `wrf_massive.base`). Each `Simulation` typically includes:
- `begin` / `end` — Simulation usable data period (ISO8601 strings or datetimes). The pipeline may add a warmup period automatically.
- `warmup_h` — Number of hours added before `begin` as model warmup.
- `sim_dir` — Directory name to hold simulation input/output.
- `settings` — Dictionary of settings that map into the namelist templates (e.g., physics options).

In the example, `simulations.py` defines `sim_test`, a 1-day test simulation. You can add additional `Simulation` objects or a list of them. The CLI command `init_sims` will write the simulation directory and render template files into it.

## Environment files (`env.yaml`)

The pipeline and CLI use an `env.yaml` file in the working directory to select machine-specific behaviour. Example files are provided:
- `env_dev.yaml` — For local development. Points `wps_tmpl_dir` and `wrf_tmpl_dir` at local submodules so the templates are available.
- `env_hpc.yaml` — Example settings for running on an HPC system (adjust template and path variables to your site).

Before using the CLI, copy one of the sample env files to `env.yaml`:

```bash
cd workspaces/example
cp env_dev.yaml env.yaml
```

`env.yaml` typically contains at least:
- `machine` — A short name used in logic inside `pipeline.py` (e.g., `dev`, `snellius`, `turbulence`).
- `wps` and `wrf` groups — Hold template directory paths and other tool-specific settings.
- `pipeline` — Name of the `Pipeline` variable defined in `pipeline.py` that should be used (e.g., `p_default` or `p_hpc`).

## Using the CLI (examples)
This repository provides a small wrapper script `cli.py` in this directory that:
- loads `env.yaml`
- imports `pipeline.py`
- selects the pipeline variable named by `env.yaml` and builds the CLI using the pipeline object

Run the wrapper script from this directory (so relative imports and templates resolve):

1) Prepare env.yaml (see above)

2) Create the simulation directory and configuration files

```bash
python3 cli.py init_sims simulations.py sim_test
```

This loads the variable `sim_test` from `simulations.py` and creates/updates the simulation directory and rendered namelists.

3) Run stages locally

Run one or more stages (comma-separated) for one or more simulation directories. If `--stages` is omitted, all stages in the pipeline are run.

```bash
# Run all stages for the single test simulation
python3 cli.py run ./test_1

# Run only the WPS stage
python3 cli.py run --stages wps ./test_1
```

4) Submit to SLURM

To submit a stage as a SLURM job, provide a jobfile (an sbatch script). Example job scripts are included in this folder (for this workspace the script is `slurm_hpc.sh`).

```bash
# Submit full pipeline stages sequentially (each will be sbatch-ed)
python3 cli.py submit --jobfile slurm_hpc.sh ./test_1

# Submit only the WRF stage and require the job to depend on job 12345
python3 cli.py submit --stages wrf --jobfile slurm_hpc.sh --dep-job 12345 ./test_1
```

The jobfile passed to `submit` must be an executable sbatch script that invokes the package's run wrapper (i.e., a small script that calls the `wrf_massive` stage runner and uses the stage name and sim dir arguments). Example job scripts are provided in the directory.


## Quick Test Run

To quickly test that your setup works, you can run a short simulation with the following commands using the provided `sim_test` (renders into `test_1`):

```bash
# Initialize the provided simulation
python3 cli.py init_sims simulations.py sim_test

# Run the simulation locally (all stages)
python3 cli.py run ./test_1

# Or submit the simulation to SLURM
python3 cli.py submit --jobfile slurm_hpc.sh ./test_1
```

Check the output logs in the `test_1` directory to verify that the simulation ran successfully.
