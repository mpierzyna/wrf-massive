<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Developer Guide

The `wrf_massive` package provides a small framework to describe and run WRF-related processing as a sequence of stages 
(a pipeline) executed for a single `Simulation`. This developer guide explains how the wrf_massive package is organised 
(focusing on `wrf_massive/base.py`) and how to implement new pipeline stages. It summarises the core classes, validators 
and conventions used by pipelines and simulations in this project.


## Core concepts
- **Simulation**: a data/config object representing a single WRF simulation (begin/end, warmup, settings, sim_dir).
- **Stage**: an atomic step in the processing pipeline. Each Stage handles its own setup and run behavior.
- **Pipeline**: an ordered collection of named Stage instances. Pipelines call each Stage in order for a target Simulation.
- **Resources**: optional resource requirements attached to a Stage (useful for job submission wrappers).

Key files:
- `wrf_massive/base.py` — core logic and types (Resources, Stage, Simulation, Pipeline).
- `wrf_massive/stages/` — the built-in stages, see [Built-in stages](#built-in-stages).
- `wrf_massive/cli.py`, `wrf_massive/workspace.py` — the `wrf-massive` command and workspace handling, see
  [Workspaces](#workspaces).
- `wrf_massive/templates/workspace/` — the files `wrf-massive init` copies into a new workspace.
- `wrf_massive/plotting.py`, `wrf_massive/third_party/` — domain plots, see [Plotting domains](#plotting-domains).
- `wrf_massive/log.py`, `wrf_massive/config.py`, `wrf_massive/tmp_dir.py` — logging, YAML (de)serialisation and the
  temporary-storage mechanism.


### Simulation
Details about individual simulations are captured in the `Simulation` class, a `pydantic.BaseModel` subclass.
- Fields: `begin`, `end` (both accept datetimes or ISO strings) bound the usable period, `warmup_h` (default 12)
  adds a spin-up window before it, `settings` (`Dict[str, str]`) is inserted into the namelist templates, `sim_dir`
  (TPath) is where everything for this run lives, and the optional `area` (`BBox`, north/west/south/east) is the
  lat/lon box of CDS forcing downloads.
- `begin_w_warmup` property returns `begin - warmup_h hours`; `name` is the name of `sim_dir`, used e.g. for SLURM
  job names and temporary-storage paths.
- `to_disk(root)` writes `simulation.yaml` into `root / sim_dir` (without the `sim_dir` field itself), and
  `from_disk(sim_dir)` reads it back. `wrf-massive init-sims` does the former for the simulations of a workspace,
  the other commands the latter.

### Stages
`Stage` is a `pydantic.BaseModel` subclass and an abstract base class. A typical Stage model includes at least:
- `work_dir` (TPath) — directory for stage-specific files. If relative, it will be interpreted relative to `Simulation.sim_dir`.
- `resources` (optional) — a `Resources` object describing job resource needs.
- `tmp_work_root` (optional) — run this stage on fast temporary storage, see "Running a stage on fast temporary
  storage" below. Off by default and invisible to the stage implementation.
- `name` — the class name, used in log messages.

To implement a new Stage, subclass `Stage` and implement the following abstract methods:
- `is_setup(self, s: Simulation) -> bool` — return True if setup is complete.
- `setup(self, s: Simulation)` — perform/setup steps (e.g. download input files).
- `is_done(self, s: Simulation) -> bool` — return True if stage completed successfully (e.g. output files exist).
- `run(self, s: Simulation)` — execute the stage (may submit jobs, call binaries, run Python tasks).

### Running a stage on fast temporary storage
Some stages are much faster if they work on a ramdisk, a node-local SSD or a scratch filesystem instead of the
(often network-mounted) simulation directory. Any stage can do this — just set `tmp_work_root`:

```python
    WRFStage(work_dir="3_wrf", tmp_work_root="/scratch-shared/me", ...)
```

While the stage runs, its work directory lives in `<tmp_work_root>/<simulation name>/<work_dir>` and the usual
location inside `sim_dir` is a symlink pointing there. Results are moved back when the stage finishes. **You do not
have to change anything in your stage implementation**: `get_work_dir()` keeps returning the same path as always,
and the filesystem takes care of the redirection.

Two options control what comes back:
- `tmp_teardown_globs` — move back only files matching these glob patterns and leave the rest behind, e.g.
  `["namelist.input", "*.log"]` to keep bulky output on scratch.
- `tmp_skip_teardown` — don't move anything back; the work dir stays on temporary storage (useful for debugging,
  or when the "temporary" storage is a shared scratch filesystem you want to keep working on).

If a stage fails, its work directory is deliberately left on temporary storage and the log says where. Re-running
picks up where it left off. If the temporary storage was wiped in the meantime, the leftover symlink is detected
and cleaned up automatically, and the stage starts from scratch.

Whenever results are still on temporary storage — after `tmp_skip_teardown`, after a failure, or for a stage that
was already finished — you can fetch them back explicitly with the `teardown` command, which takes the same
`-s/--stages` selection as `run`:

```bash
wrf-massive teardown -s wrf ./test_1         # move the wrf work dir back into the simulation dir
wrf-massive teardown --all ./test_1          # all stages, and ignore tmp_teardown_globs
```

Stages that are not on temporary storage are simply skipped, so running it over the whole pipeline is safe.

Two things are refused, because they would relocate far more than intended: an absolute `work_dir`, and a
`work_dir` that is (or contains) the simulation directory itself — such as `MarkDone(work_dir=".")`. Inside a
`StageArray` such a substage is simply skipped with a warning; configured directly on a stage it raises an error.

Note that temporary storage is only used when the stage is run through a `Pipeline` (or a `StageArray`). Calling
`stage.run(s)` by hand does not relocate anything. And because the redirection works through a symlink at the
original path, avoid calling `.resolve()` on the work dir or storing absolute paths when you construct a stage —
you would see temporary paths leak into your output.

### Grouping stages: `StageArray`
`StageArray` runs several stages inside a single job, which is useful on HPC where one allocation should cover
e.g. WRF plus its postprocessing. The array is submitted as one job, so its `resources` replace those of its
substages (with a warning if they differ). Setting `tmp_work_root` on the array applies to all of its substages; substages
without one can also carry their own `tmp_work_root` to be relocated individually.

Whatever is relocated is moved back only once the *last* substage has finished — so a later substage can read an
earlier one's output while both are still on fast storage (this is the point of putting WRF and its postprocessing
in one array). Options like `tmp_teardown_globs` and `tmp_skip_teardown` are set on the individual substages, and
`teardown` on the array reaches all of its substages.

### Resources
`Resources` is a pydantic model with:
- `n_tasks`, `cpus_per_task`, `mem_per_cpu`, and optional `walltime` (`datetime.timedelta`).
- `n_nodes` — number of nodes the job runs on (default 1). `n_tasks` is the total over all nodes, e.g.
  `Resources(n_nodes=2, n_tasks=256, cpus_per_task=1, mem_per_cpu="1G")` runs WRF with 256 MPI tasks on 2 nodes.
  SLURM decides how many tasks go on each node (it fills nodes one after the other), so the split is only even if
  the tasks exactly fill the nodes, e.g. 2 × 128 on nodes with 128 cores.
  Only MPI programs (WRF, WPS) can make use of several nodes; Python stages such as postprocessing run on one.
- `cpus_total` is a convenience property computed as `n_tasks * cpus_per_task`.
- `max_concurrent` (optional `int`) — limits how many jobs for this stage are allowed to run at the
  same time. Unset by default (no limit). See "Submitting to SLURM" below for when to use it.

### Pipeline
- Construct a pipeline with named Stage instances: `Pipeline(foo=StageA(...), bar=StageB(...))`.
  The argument names are the keys used to reference stages (`p["foo"]`, `p.stage_names`, `--stages` in the CLI).
- `Pipeline.run(s)` executes all stages in order, or a subset via `stages=` (a list or a single name). It skips the
  whole simulation if `sim_dir/.done` exists.
- `run_stage(s, name)` skips a stage if its work dir contains `.done`, or if it is already `is_setup` and `is_done`.
  Otherwise it runs `setup()` unless already set up and `run()` unless already done. `force_setup`/`force_run`
  re-run those steps anyway; a `.done` marker still skips the stage.
- `teardown(s)` / `teardown_stage(s, name)` move work dirs back from temporary storage (see above).

### Submitting to SLURM (`wrf-massive submit`)
The CLI's `submit` command submits each requested stage of each simulation as its own SLURM job. Every submitted
stage needs `resources`, which become the `sbatch` options `--nodes`, `--ntasks`, `--cpus-per-task`,
`--mem-per-cpu` and, if `walltime` is set, `--time`. The stages of a simulation depend on each other (`afterok`), and `--dep-job` makes
the first one also wait for another job. SLURM writes the job output to `<sim_dir>/slurm/<stage>_<job id>.out`.
Each simulation is submitted independently, so submitting many simulations at once can run many
jobs for the same stage in parallel.

For stages that shouldn't have too many jobs running at the same time — for example, a download
stage that would overwhelm a rate-limited remote server — set `max_concurrent` on that stage's
`Resources`. This limits how many jobs for that stage are running at once, no matter how many
simulations you submit together or how many CPUs are free.


### Helpers and conventions
`base.py` defines small helper validators used by the Pydantic models:
- `_parse_datetime(v)` — accepts either a datetime or ISO format string and returns a `datetime.datetime`.
- `_ensure_path(v)` — converts str or Path to `pathlib.Path`.
- `_ensure_path_exists(v)` — like `_ensure_path` but raises `ValueError` if the path doesn't exist.
- `_mkdir_if_not_exists(p)` — creates the directory (parents=True) and returns the Path.

These are connected to annotated types using Pydantic's `BeforeValidator`:
- `TPath` — Path or str; converted to `pathlib.Path`.
- `TPathExists` — same but validated to already exist.
- `TPathMkdir` — path that will be created if missing.

This means: declare fields using these alias types in Pydantic models to get consistent path handling.

Additionally,
- Use `stage.get_work_dir(s)` to obtain the directory for stage files; if `work_dir` on the stage is relative, it is
  resolved relative to `s.sim_dir`. By default this creates the directory.
- Stages and Pipelines use a simple `.done` marker convention: if a stage's work directory contains a file named
  `.done`, the pipeline's `run_stage` will skip the stage, even with `force_run`. `MarkDone` writes it (with
  `work_dir="."` for the whole simulation), and `GarbageCollectStage` writes it into a stage's work dir after
  deleting that stage's data, so it isn't run again.
- To skip the entire simulation, create a `.done` file in `s.sim_dir` (the Pipeline checks for this as well).

## Built-in stages

| Stage | Module | Purpose |
|---|---|---|
| `PullCdsStage` | `stages/forcing/cds.py` | Download forcing (CERRA, ERA5) from the Copernicus Climate Data Store, split per time step for WPS. |
| `PullCerraStage` | `stages/forcing/cerra.py` | Copy pre-mirrored CERRA files from an rclone remote. |
| `WPSStage` | `stages/wps/wps.py` | Render the WPS namelists and run geogrid, ungrib (CERRA and ERA5) and metgrid. |
| `WRFStage` | `stages/wrf/wrf.py` | Render `namelist.input`, link the `met_em` files and run WRF. |
| `PostProcStage`, `Cn2PostProcStage` | `stages/postproc/` | Extract variables from `wrfout` files and compute e.g. $C_n^2$ (NetCDF output). |
| `MarkDone` | `stages/misc/done.py` | Write a `.done` marker, e.g. for the whole simulation. |
| `GarbageCollectStage` | `stages/misc/gc.py` | Delete data of an earlier stage once it is no longer needed (only when `armed`). |
| `StageArray` | `stages/misc/array.py` | Run several stages in one job, see above. |

Notes:
- `WRFStage` checks `resources.n_tasks` against the range recommended for the domain size in the namelist
  (`e_we`/`e_sn`, following the [WRF forum](https://forum.mmm.ucar.edu/threads/choosing-an-appropriate-number-of-processors.5082/)):
  the range is logged as info, and a warning is logged if `n_tasks` is below or above it.
- The postprocessing stages need `wrf-python`, which is installed with the `postproc` extra of wrf-massive.

See `stages/README.md` for a short overview of the four pipeline steps.

## Workspaces

A workspace holds everything specific to one study. Create one with `wrf-massive init <dir>` (default: current
directory); existing files are only overwritten with `--force`. The template contains:

| File | Purpose |
|---|---|
| `env.yaml` | Machine-specific settings: paths to WPS/WRF, which pipeline to use, log level. |
| `pipeline.py` | Assembles stages into named pipelines (`p_default`: CERRA + ERA5 from CDS, WPS, WRF, `MarkDone`). |
| `simulations.py` | Defines `Simulation` objects (period, warmup, namelist settings, sim dir, area). |
| `namelist.tmpl.wps`, `namelist.tmpl.input` | Jinja2 templates for the WPS/WRF namelists, incl. the domain. |
| `myoutfields.txt` | Additional WRF output fields. |
| `slurm.job.sh` | Job script for `wrf-massive submit`. |
| `.gitignore` | Ignores SLURM output, forcing files and the example simulation. |

The template is a starting point and does not run as is: set the WPS/WRF paths in `env.yaml`, set up CDS credentials
(`~/.cdsapirc` or `CDSAPI_URL`/`CDSAPI_KEY`) and adapt the domain and simulations to your study.

### The env file
The env file contains the settings that differ between machines:
- `pipeline` (required) — name of the pipeline variable in `pipeline.py` to use, e.g. `p_default`.
- `log_level` (optional) — `debug`, `info` (default), `warning`, `error` or `critical`.
- anything else your `pipeline.py` needs, e.g. the `wps` and `wrf` groups of the template.

The directory containing the env file is the workspace root. The CLI looks for the env file in the current
directory and its parents, so you can run it from anywhere inside the workspace, e.g. from a simulation dir.
Relative paths in `pipeline.py` and in the env file are relative to the workspace root; paths given on the command
line are relative to where you run the command.

To work on several machines, keep one env file per machine (e.g. `env_laptop.yaml`, `env_hpc.yaml`) and select it with
`--env-file`/`-e` or the `WRF_MASSIVE_ENV_FILE` environment variable (`env.yaml` by default):

```bash
wrf-massive -e env_hpc.yaml submit -j slurm.job.sh test_1
```

Alternatively, symlink the active one to `env.yaml` (`ln -sf env_hpc.yaml env.yaml`).

In `pipeline.py`, read the settings with `load_env()`, which returns the env file the CLI was started with:

```python
from wrf_massive.workspace import load_env

env = load_env()
wps = WPSStage(work_dir="2_wps", **env["wps"], ...)
```

### Commands

```bash
wrf-massive init my_study                        # create a new workspace
wrf-massive init-sims simulations.py sim_test   # create simulation dirs from variable `sim_test`
wrf-massive run test_1                           # run all stages locally
wrf-massive run --stages wps,wrf test_1          # run a subset of stages
wrf-massive teardown --stages wrf test_1         # move results back from temporary storage
wrf-massive submit -j slurm.job.sh test_1        # submit each stage as its own SLURM job
wrf-massive --debug submit -j slurm.job.sh test_1   # print the sbatch commands instead of submitting
```

`init-sims` creates the simulation dirs in the workspace root unless `--root` is given. `submit` calls the job script
with the stage name and simulation dir as arguments, so the script should run `wrf-massive run --stages=$1 $2`. The
jobs use the same env file as the `submit` call.

### Plotting domains
`wrf-massive plot` plots WPS domains, either from the namelist before running geogrid or from the geogrid output.
It works without a workspace:

```bash
wrf-massive plot domain-nml namelist.tmpl.wps            # expected domain layout from a (template) namelist.wps
wrf-massive plot domain-nml namelist.tmpl.wps --info 15  # further options of plot_domains.py, see --help
wrf-massive plot domain-geo-em 2_wps/geo_em.d01.nc        # terrain of the geogrid output
```

`domain-nml` runs [`plot_domains.py`](https://github.com/wrf-model/WPS/blob/release-v4.7.0/util/plot_domains.py)
from WPS v4.7.0 by Anthony Islas (2025), included unchanged in `wrf_massive/third_party/`, except for the added
copyright information. It saves the plot to `--save_file` (default `domains.png`) when no interactive display is
available, and opens a window otherwise. `domain-geo-em` saves the plot next to the file as `<name>_domains.png`.

## Implementing a new Stage

1) Subclass `Stage` and implement the abstract methods. Example minimal skeleton:

```python
    class MyStage(Stage):
        work_dir: TPath = pathlib.Path("mystage")  # relative -> under sim_dir. Defaults are not validated, so no str!

        def is_setup(self, s: Simulation) -> bool:
            # check presence of staged input, or config files
            return (self.get_work_dir(s) / "input.ok").exists()

        def setup(self, s: Simulation):
            work = self.get_work_dir(s)
            # prepare input files, download, symlink, templates, etc.
            # write markers if needed

        def is_done(self, s: Simulation) -> bool:
            # check expected output files
            return (self.get_work_dir(s) / "result.nc").exists()

        def run(self, s: Simulation):
            work = self.get_work_dir(s)
            # run the work (call executable or Python code); is_done() then finds its output
```

   Let `is_done()` check the stage's output instead of writing a `.done` marker in `run()`: a marker makes the
   pipeline skip the stage for good, even with `--force-run`.

2) Keep side-effects contained in the stage work directory when possible. Relative `work_dir` paths are recommended to 
   keep simulation folders self-contained.
3) Use `Simulation.settings` to pass parameters to stages; `settings` is a simple `Dict[str,str]` intended for 
   templating namelists etc.
4) Prefer idempotent `setup()` and `run()` implementations so re-running does not corrupt state; 
   rely on `is_setup` / `is_done` for quick checks.
5) If a stage requires an absolute path (shared data, central cache), set `work_dir` to an absolute path.
   `Stage.get_work_dir` returns it as is (without creating it), and such a stage cannot run on temporary storage.

## Registering stages in a pipeline

Pipeline creation example:

```python
p = Pipeline(prep=PrepStage(work_dir="prep"), runwrf=RunWRFStage(work_dir="wrf"))
p.run(sim)
```

When calling `p.run`, if `sim.sim_dir/.done` exists the whole simulation is skipped.
If a stage's work directory contains `.done` the pipeline will skip that stage.

In a workspace, define pipelines as module-level variables in `pipeline.py` (e.g. `p_default`, `p_hpc`); the
`pipeline` key of the env file selects the one the CLI uses.


## Tips
- Keep `is_setup`/`is_done` checks fast and deterministic so tests can assert stage behavior without expensive I/O when possible.
- When adding a stage, add unit tests to `tests/` (run them with `uv run pytest`) covering:
  - `is_setup` false -> `setup()` creates required files
  - `run()` produces expected outputs and `.done` markers
  - `get_work_dir()` behavior with relative and absolute `work_dir` values

  `tests/fixtures.py` has small test stages (`StageA`, `StageB`, ...) and a `simple_simulation` fixture to test
  pipeline behaviour without WRF/WPS.


## Coding style & best practices
- Use pydantic models for config fields and validation when possible.
- Keep external side-effects (network, large downloads) behind `setup()` so tests can patch/mock them.
- Be explicit about file names and markers to avoid accidental collisions between stages.
- Use black to format code (`uv run black src tests`, configured in `pyproject.toml`); files in `third_party/` are
  vendored and excluded.

