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
- `wrf_massive/cli.py`, `wrf_massive/log.py`, and other modules integrate with the framework (logging, CLI); this 
  README focuses on the developer-facing API in `base.py`.


### Simulation
Details about individual simulations are captured in the `Simulation` class, a `pydantic.BaseModel` subclass.
- Fields: `begin`, `end` (both accept datetimes or ISO strings), `warmup_h`, `settings` (dict), and `sim_dir` (TPath).
- `begin_w_warmup` property returns `begin - warmup_h hours`.
- `to_disk(root)` writes a `simulation.yaml` to the chosen root / sim_dir (excluding the `sim_dir` field itself in the 
  YAML) so simulation configs can be reloaded.
- `from_disk(sim_dir)` reads `simulation.yaml` and returns a `Simulation` object; it expects the directory to exist.

Simulation persistence:
- Use `Simulation.to_disk(root)` to write `simulation.yaml` into `root / sim_dir`.
- Use `Simulation.from_disk(sim_dir)` to reconstruct a Simulation object from disk;
  it expects `simulation.yaml` to be present.

### Stages
`Stage` is a `pydantic.BaseModel` subclass and an abstract base class. A typical Stage model includes at least:
- `work_dir` (TPath) — directory for stage-specific files. If relative, it will be interpreted relative to `Simulation.sim_dir`.
- `resources` (optional) — a `Resources` object describing job resource needs.
- `tmp_work_root` (optional) — run this stage on fast temporary storage, see "Running a stage on fast temporary
  storage" below. Off by default and invisible to the stage implementation.

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

Two things are refused, because they would relocate far more than intended: an absolute `work_dir`, and a
`work_dir` that is (or contains) the simulation directory itself — such as `MarkDone(work_dir=".")`. Inside a
`StageArray` such a substage is simply skipped with a warning; configured directly on a stage it raises an error.

Note that temporary storage is only used when the stage is run through a `Pipeline` (or a `StageArray`). Calling
`stage.run(s)` by hand does not relocate anything. And because the redirection works through a symlink at the
original path, avoid calling `.resolve()` on the work dir or storing absolute paths when you construct a stage —
you would see temporary paths leak into your output.

### Grouping stages: `StageArray`
`StageArray` runs several stages inside a single job, which is useful on HPC where one allocation should cover
e.g. WRF plus its postprocessing. Setting `tmp_work_root` on the array applies to all of its substages; substages
without one can also carry their own `tmp_work_root` to be relocated individually.

Whatever is relocated is moved back only once the *last* substage has finished — so a later substage can read an
earlier one's output while both are still on fast storage (this is the point of putting WRF and its postprocessing
in one array). Options like `tmp_teardown_globs` and `tmp_skip_teardown` are set on the individual substages.

### Resources
`Resources` is a pydantic model with:
- `n_tasks`, `cpus_per_task`, `mem_per_cpu`, and optional `walltime` (`datetime.timedelta`).
- `cpus_total` is a convenience property computed as `n_tasks * cpus_per_task`.
- `max_concurrent` (optional `int`) — limits how many jobs for this stage are allowed to run at the
  same time. Unset by default (no limit). See "Submitting to SLURM" below for when to use it.

### Pipeline
- Construct a pipeline with named Stage instances: Pipeline(foo=StageA(...), bar=StageB(...)). 
  The argument names are the keys used to reference stages.
- Use `Pipeline.run(s)` to execute all stages (or pass a subset via `stages=` or a single stage name).
- `run_stage` supports `force_setup` and `force_run` flags to re-run setup or run even when checks say done.

### Submitting to SLURM (`cli.py submit`)
The CLI's `submit` command submits each requested stage of each simulation as its own SLURM job.
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
  `.done`, the pipeline's `run_stage` will skip the stage.
- To skip the entire simulation, create a `.done` file in `s.sim_dir` (the Pipeline checks for this as well).

## Implementing a new Stage

1) Subclass `Stage` and implement the abstract methods. Example minimal skeleton:

```python
    class MyStage(Stage):
        work_dir: TPath = "mystage"  # relative -> under sim_dir
        resources: Resources | None = None

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
            # run the work (call executable or Python code)
            # on success, optionally create a .done file in work
            (work / ".done").write_text("ok")
```

2) Keep side-effects contained in the stage work directory when possible. Relative `work_dir` paths are recommended to 
   keep simulation folders self-contained.
3) Use `Simulation.settings` to pass parameters to stages; `settings` is a simple `Dict[str,str]` intended for 
   templating namelists etc.
4) Prefer idempotent `setup()` and `run()` implementations so re-running does not corrupt state; 
   rely on `is_setup` / `is_done` for quick checks.
5) If a stage requires an absolute path (shared data, central cache), set `work_dir` to an absolute path. 
   `Stage.get_work_dir` will not try to create a directory under `sim_dir` when `work_dir` is absolute.

## Registering stages in a pipeline

Pipeline creation example:

```python
    p = Pipeline(prep=PrepStage(work_dir="prep"), runwrf=RunWRFStage(work_dir="wrf"))
    p.run(sim)

```

When calling `p.run`, if `sim.sim_dir/.done` exists the whole simulation is skipped. 
If a stage's work directory contains `.done` the pipeline will skip that stage.


## Tips
- Keep `is_setup`/`is_done` checks fast and deterministic so tests can assert stage behavior without expensive I/O when possible.
- When adding a stage, add unit tests to `tests/` covering:
  - `is_setup` false -> `setup()` creates required files
  - `run()` produces expected outputs and `.done` markers
  - `get_work_dir()` behavior with relative and absolute `work_dir` values


## Coding style & best practices
- Use pydantic models for config fields and validation when possible.
- Keep external side-effects (network, large downloads) behind `setup()` so tests can patch/mock them.
- Be explicit about file names and markers to avoid accidental collisions between stages.
- Use black to format code to maintain consistency.

