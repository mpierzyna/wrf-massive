<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# WRF Massive

This project provides tooling to orchestrate large-scale WRF (Weather Research and Forecasting) simulation using Slurm. 
The main goal is to make it easy to define, stage, submit, and monitor many WRF runs (including pre/post processing steps).
The framework is designed to be modular, which allows you to test the entire pipeline on a laptop or workstation and
scale to HPC clusters with minimal changes.

Key points:
- Orchestration: the code manages job creation, staging of input/output, submission to Slurm.
- Extensible: the project is structured so that general-purpose code lives in the `wrf_massive` package, while
  project-specific configuration and templates live in separate workspaces.

More specifically,
- The orchestration code and core CLI are implemented in `wrf_massive`.
  - `wrf_massive` is designed to be as project/workspace-agnostic as possible. 
    It contains the main logic to configure runs, produce Slurm job scripts, submit them and collect outputs.
- Concrete projects/simulation runs live in separate workspaces, which `wrf-massive init` creates from a template
  with sample configuration, namelist templates and a job script.
- Optionally, WRF and WPS are included as submodules in `submodules`. 
  You can pull them using `git submodule update --init --recursive` and build them according to their own instructions.

> [!NOTE]
> WRF Massive was created to generate the WRF-based $C_n^2$ training data of the
> [OTProf](https://github.com/mpierzyna/otprof) deep learning project (1 year of CERRA-forced simulations for the
> Netherlands and the Canary Islands). The repository [`otprof-wrf`](https://github.com/mpierzyna/otprof-wrf) contains
> these production workspaces and shows how wrf-massive is used at scale. For a template, which needs modification for
> your own setup, create a workspace with `wrf-massive init`.

## Submodules vs external WRF/WPS

- Using the included submodules (optional) to build WRF/WPS inside this repository:
  - Clone the NCAR repos using `git submodule update --init --recursive`
  - Compile and setup WRF as per NCAR's instructions.

- Using an external WRF/WPS installation (recommended for reproducible or managed HPC installs):
  - If you already have a system-wide or separately-maintained WRF/WPS installation, you do not need the submodules. 
    Point this orchestration tool to your existing installation in your workspace's `env.yaml`. This repository only
    orchestrates runs and does not require WRF/WPS to be compiled inside it.


## Workspaces

A workspace is a directory with the configuration of one study: an env file with machine-specific settings, the
pipeline, the simulations and the WPS/WRF namelist templates. Create one with

```bash
wrf-massive init my_study
```

and see the [developer guide](src/wrf_massive/README.md#workspaces) for what the files do.

## Environment setup

This package is managed with the uv package manager. All dependencies are listed in `pyproject.toml`.
To use wrf-python, a functional `gfortran` compiler needs to be installed (typically by having a working gcc installation).
Set up the environment by running

```bash
uv sync
```

Alternatively, just running commands with `uv run` will always make sure the environment is up to date.


## Quick start (high level)

1. Prepare WRF/WPS.
2. Create a workspace with `wrf-massive init my_study` and adapt `env.yaml`, `pipeline.py`, `simulations.py` and the
   namelist templates to your study.
3. Create, run and submit simulations with the `wrf-massive` CLI from anywhere inside the workspace
   (`wrf-massive --help` lists the commands):

   ```bash
   wrf-massive init-sims simulations.py sim_test   # create simulation dir test_1
   wrf-massive run test_1                           # run all stages locally
   wrf-massive submit -j slurm.job.sh test_1        # or submit each stage as SLURM job
   ```

## How to cite
When using wrf-massive, please cite it either using the `./CITATION.cff` file or the following information:

> Pierzyna, M. (2026). *wrf-massive: flexible, scalable orchestration of WRF simulations* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24113829
