<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# WRF Massive

WRF Massive orchestrates large numbers of WRF (Weather Research and Forecasting) simulations, from downloading the
forcing data through WPS and WRF to postprocessing, on a laptop or on an HPC cluster with SLURM. You describe the
simulations and the steps they go through in a small Python project (a *workspace*), and the `wrf-massive` command
creates, runs and submits them. The same workspace can be tested locally and then scaled to a cluster by changing
only machine-specific settings.

Features:
- **Stages**: each step in the WRF processing chain is a `Stage` defined in Python: forcing download, WPS, WRF, 
  and postprocessing (e.g. $C_n^2$).
- **Pipelines**: pipelines define the order in which stages are executed to process a simulation. They can differ per machine.
- **Workspaces**: different WRF studies (e.g., different domains) can be kept isolated in different workspaces, 
  each with their own pipelines and stages.
- **SLURM submission**: every stage of every simulation becomes a SLURM job with its own resources (nodes, tasks,
  memory, walltime), chained by dependencies. Concurrency can be capped per stage, e.g. for downloads.
- **Restartable**: finished stages are marked as done and skipped on re-runs.
- **Monitoring**: optionally log the memory and disk usage of each stage to CSV files, also within SLURM jobs.
- **Fast storage**: stages can run in temporary directories, e.g., a ramdisk or scratch filesystem. 
  Results are moved back automatically when the stage finishes.

> [!NOTE]
> WRF Massive was created to generate the WRF-based $C_n^2$ training data of the
> [OTProf](https://github.com/mpierzyna/otprof) deep learning project (1 year of CERRA-forced simulations for the
> Netherlands and the Canary Islands). The repository [`otprof-wrf`](https://github.com/mpierzyna/otprof-wrf) contains
> these production workspaces and shows how wrf-massive is used at scale.

## Requirements

- Python 3.12 or newer.
- Only for the postprocessing stages (e.g. $C_n^2$): a Fortran compiler (e.g. `gfortran` from a working gcc
  installation), to build `wrf-python` from source. It is installed with the optional `postproc` extra.
- Compiled WPS and WRF, and the WPS static geographical data. wrf-massive orchestrates them but does not build them;
  see [WRF and WPS](#wrf-and-wps).
- For the forcing data: a [CDS account](https://cds.climate.copernicus.eu/how-to-api) with its credentials in
  `~/.cdsapirc`, or an [rclone](https://rclone.org) remote with mirrored CERRA files.
- Optional: SLURM, to schedule pipeline stages with assigned ressources. Without SLURM, each stage just executes
  sequentially in the terminal, using all available ressources.

## Installation

wrf-massive is installed from GitHub into your own project, at a release tag.

### With uv (recommended)

```bash
uv init my_study && cd my_study   # new project, unless you already have one
uv add "wrf-massive @ git+https://github.com/mpierzyna/wrf-massive@v2.0.0"
uv run wrf-massive --help
```

To use the postprocessing stages, add wrf-massive with the `postproc` extra instead. uv then also installs the
`wrf-python` version that wrf-massive pins (a git tag of NCAR's repository):

```bash
uv add "wrf-massive[postproc] @ git+https://github.com/mpierzyna/wrf-massive@v2.0.0"
```

Prefix the commands below with `uv run`, or activate the environment with `source .venv/bin/activate`.

### With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install "wrf-massive @ git+https://github.com/mpierzyna/wrf-massive@v2.0.0"
wrf-massive --help
```

With the postprocessing stages, install `wrf-python` from its git tag explicitly, because pip would otherwise use the
older release on PyPI:

```bash
pip install "wrf-python @ git+https://github.com/NCAR/wrf-python@v1.4.2" \
            "wrf-massive[postproc] @ git+https://github.com/mpierzyna/wrf-massive@v2.0.0"
```

### As a git submodule

To work on wrf-massive alongside your project (as [`otprof-wrf`](https://github.com/mpierzyna/otprof-wrf) does), add
it as a submodule and install it in editable mode:

```bash
git submodule add https://github.com/mpierzyna/wrf-massive submodules/wrf-massive
git -C submodules/wrf-massive checkout v2.0.0
uv add --editable ./submodules/wrf-massive              # or with --extra postproc
```

## Getting started

1. Create a workspace in your project, here in the directory `my_domain`:

   ```bash
   wrf-massive init my_domain && cd my_domain
   ```

2. Adapt it to your study: the WPS/WRF paths in `env_local.yaml` (`env.yaml` links to it), the domain in `namelist.tmpl.wps` and
   `namelist.tmpl.input`, the pipeline in `pipeline.py` and the simulations in `simulations.py`. Check the domain
   before running anything:

   ```bash
   wrf-massive plot domain-nml namelist.tmpl.wps   # saves domains.png
   ```

3. Create the simulation directories, then run them locally or submit them to SLURM:

   ```bash
   wrf-massive init-sims simulations.py sim_test   # create simulation dir test_1
   wrf-massive run test_1                           # run all stages locally
   wrf-massive submit test_1                        # or submit each stage as SLURM job
   ```

`wrf-massive --help` lists all commands, and `wrf-massive <command> --help` their options.

## Workspaces

A workspace is a directory with the configuration of one study: an env file (`env.yaml`) with machine-specific
settings, the pipeline, the simulations and the WPS/WRF namelist templates. The directory containing `env.yaml` is the
workspace root, so the CLI works from any subdirectory. To work on several machines, keep one env file per machine and
symlink the one of the current machine to `env.yaml` (`ln -sf env_hpc.yaml env.yaml`).

See the [developer guide](src/wrf_massive/README.md#workspaces) for what the files do and how to write your own
stages.

## WRF and WPS

wrf-massive does not compile WRF or WPS. Point your workspace's `env.yaml` to existing installations, e.g. a
system-wide or module-provided install on your cluster (recommended for reproducible HPC runs) or your own build
following NCAR's instructions.

For development, this repository includes WRF and WPS as optional submodules in `submodules/`. Pull them with
`git submodule update --init --recursive` and build them according to their own instructions.

## Development

```bash
git clone https://github.com/mpierzyna/wrf-massive && cd wrf-massive
uv sync                   # install the package and its development dependencies
uv sync --extra postproc  # ... including wrf-python for the postprocessing stages (needs a Fortran compiler)
uv run pytest             # run the tests (the postprocessing tests are skipped without wrf-python)
```

Releases are made with `uvx bump-my-version bump patch|minor|major`, which updates the version, the citation
metadata and the install instructions in this README, then commits and tags.

## How to cite
When using wrf-massive, please cite it either using the `./CITATION.cff` file or the following information:

> Pierzyna, M. (2026). *wrf-massive: flexible, scalable orchestration of WRF simulations* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24113829
