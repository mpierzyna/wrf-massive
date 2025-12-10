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
- Concrete projects/simulation runs should live in `workspaces`
  - An example workspace is provided in `workspace/example` with sample configuration, job templates and instructions.
- Optionally, WRF and WPS are included as submodules in `submodules`. 
  You can pull them using `git submodule update --init --recursive` and build them according to their own instructions.

## Submodules vs external WRF/WPS

- Using the included submodules (optional) to build WRF/WPS inside this repository:
  - Clone the NCAR repos using `git submodule update --init --recursive`
  - Compile and setup WRF as per NCAR's instructions.

- Using an external WRF/WPS installation (recommended for reproducible or managed HPC installs):
  - If you already have a system-wide or separately-maintained WRF/WPS installation, you do not need the submodules. 
    Point this orchestration tool to your existing installation using the configuration mechanism in your workspace 
    (see `workspace/example/README.md` for an example configuration). This repository only orchestrates runs and does 
    not require WRF/WPS to be compiled inside it.


## Workspace examples

The repository contains a `workspace/example` folder with a self-contained example workspace and an example `README.md`. 
Use that as your primary reference for how to structure a workspace that uses this tool. 

## Environment setup

A conda environment file is provided as `environment.yml` at the project root to create a reproducible Python environment. 
It contains the Python packages typically needed to run the orchestration tools, utility scripts and tests.

Create the environment (conda/miniconda/anaconda required):

```bash
conda env create -f environment.yml
# or to update an existing env
conda env update -f environment.yml
```

Activate it:

```bash
conda activate wrf-massive  # or the name defined in environment.yml
```

If you prefer pip or another environment manager, use the package list in `environment.yml` as a reference to recreate 
the environment. The project also contains `pyproject.toml` files that provide metadata for the Python packages in the workspace.

## Quick start (high level)

1. Create and activate the Python environment using `environment.yml`.
2. Prepare WRF/WPS:
3. Inspect `workspace/example` to copy example configuration and job templates into your workspace.
4. Use the CLI in `wrf_massive` to create and submit evaluations (see `wrf_massive/cli.py` for the available commands). 
   Typical invocation is via the Python CLI in your activated environment.
