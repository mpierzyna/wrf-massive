# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

`wrf-massive` orchestrates large numbers of WRF (Weather Research and Forecasting) simulations with SLURM. It manages
staging of input data, job submission, and output collection so a simulation pipeline can be tested locally (laptop/
workstation) and then scaled to an HPC cluster with minimal changes. It was built to generate a 1-year CERRA-forced
WRF dataset for the Netherlands (see the [OTProf](https://github.com/mpierzyna/otprof) project for the production
use case).

- Package manager: `uv` (`uv sync` to install, `uv run <cmd>` to run anything in the managed env).
- Python: `>=3.12`.
- `wrf-python` needs a working `gfortran`/gcc toolchain to build.
- Formatting: `black` (line-length 120, see `pyproject.toml`); submodules are excluded from formatting.
- `README.md` files (top-level, `src/wrf_massive/`, `workspaces/example/`) are user-facing: describe
  what a feature does and how to use it, not the underlying implementation mechanics. Implementation
  details belong here in `CLAUDE.md` instead.

## Repository layout

```
src/wrf_massive/        # generic, project-agnostic orchestration package
  base.py                # Stage / Simulation / Pipeline / Resources core abstractions
  config.py              # BaseYAMLConfig: pydantic <-> yaml (de)serialization helpers
  cli.py                  # get_pipeline_cli(pipeline) -> click.Group (init_sims/run/teardown/submit)
  log.py                  # logging helpers
  tmp_dir.py              # move a stage's work dir to a fast tmp location (e.g. ramdisk) and back
  stages/
    forcing/              # download/prepare forcing data for WPS (currently: CERRA via rclone)
    wps/                  # render namelist.wps, run WPS (geogrid/ungrib/metgrid)
    wrf/                  # render namelist.input, run WRF
    postproc/             # postprocessing of WRF output (e.g. Cn2)
    misc/                 # small utility stages (array.py, done.py, gc.py)
    utils.py               # render_template, run_cmd_logged, namelist template loader/getter
workspaces/example/      # a concrete, documented example workspace (the reference for how to build your own)
  pipeline.py             # assembles Stage instances into named Pipeline(s) (p_default, p_hpc, ...)
  simulations.py          # defines Simulation objects (begin/end/warmup/settings/sim_dir)
  env_dev.yaml/env_hpc.yaml  # machine-specific paths/settings, copied to env.yaml before use
  namelist.tmpl.wps/.input   # Jinja2 templates for WPS/WRF namelists
  cli.py                  # thin wrapper: loads env.yaml, picks pipeline, calls get_pipeline_cli
submodules/WRF, submodules/WPS   # optional NCAR submodules (git submodule update --init --recursive)
tests/                    # pytest; fixtures.py has StageA/StageB/StageFail/StageReadsOther test doubles
                          #   + simple_simulation fixture
```

`wrf_massive` itself must stay workspace-agnostic — project-specific config, templates and pipelines belong in a
`workspaces/<name>` directory (see `workspaces/example` as the canonical template).

## Core abstractions (`src/wrf_massive/base.py`)

- **`Simulation`** (`pydantic` + `BaseYAMLConfig`): one WRF run. `begin`/`end` bound the *usable* period;
  `begin_w_warmup = begin - warmup_h` adds a spin-up window. `settings: Dict[str,str]` feeds Jinja2 namelist
  templates. `sim_dir` is where everything for this run lives. `area: BBox | None` is an optional lat/lon
  bounding box (N/W/S/E) used by CDS-based forcing downloads (must fully cover the WRF domain; see
  `stages/forcing/geo.py`). Persisted via `to_disk()`/`from_disk()` as `simulation.yaml`.
- **`Stage`** (abstract `pydantic.BaseModel`): one pipeline step. Every stage implements
  `is_setup`, `setup`, `is_done`, `run`. `work_dir` is relative to `sim_dir` unless given as an absolute path
  (`get_work_dir()` handles resolution + directory creation). Stages should keep side effects (network, big
  downloads) inside `setup()`/`run()` so they're easy to mock in tests, and should be idempotent.
- **Running a stage on temporary storage**: any `Stage` can have its work dir relocated to fast storage (ramdisk,
  node-local or shared scratch) by setting `tmp_work_root`. The mechanism lives entirely in
  `Stage.tmp_work_dir(s)` (a `@contextlib.contextmanager` wrapping `tmp_dir.py`'s `setup_tmp_work_dir` /
  `teardown_tmp_work_dir`) and is entered by the *driver* — `Pipeline.run_stage` wraps its whole setup/run block
  in it. Concrete stages therefore need **no** changes: the work dir is moved to
  `<tmp_work_root>/<sim.name>/<work_dir>` and the original path becomes a symlink to it, so `get_work_dir()` keeps
  returning the same path and the kernel does the redirection (this also makes cross-stage reads like
  `s.sim_dir / self.met_em_dir` follow the *other* stage's symlink for free). Knobs: `tmp_teardown_globs` (move only
  matching files back), `tmp_skip_teardown` (leave results in tmp even on success). Teardown can also be forced out
  of band via `Stage.teardown_tmp_work_dir(s, all_files=...)` / `Pipeline.teardown()` / the CLI's `teardown`: it decides
  purely from the on-disk symlink (not from `tmp_work_root`), is a no-op on a work dir that is not symlinked, and
  ignores `tmp_skip_teardown` and every `.done` marker — that is how results left on scratch by
  `tmp_skip_teardown`, by a crash, or by an already setup+done stage (which `run_stage` skips *before* relocating
  anything) are reclaimed. `StageArray` overrides it to recurse into its substages. On an exception the work dir is
  left symlinked into tmp and a re-run resumes in place (`setup_tmp_work_dir` is idempotent, and both it and
  `get_work_dir` repair a dangling symlink if the tmp root was wiped in between). `_check_tmp_work_dir_allowed`
  refuses an absolute `work_dir` or one that is/contains `sim_dir` (e.g. `MarkDone(work_dir=".")`) — that would
  relocate the entire simulation. Caveat for stage authors: the redirection is *by path*, so calling `.resolve()`
  on the work dir or caching an absolute path at construction time leaks tmp paths. A bare `stage.run(s)` called
  outside `Pipeline`/`StageArray` gets no tmp dir.
- **`Pipeline`**: ordered dict of named `Stage`s. `run_stage()` skips a stage if its `work_dir/.done` exists or if it
  is already both `is_setup()` and `is_done()` (checked before any tmp relocation, so a finished simulation costs
  nothing), then calls `setup()` unless already `is_setup()` and `run()` unless already `is_done()`, all inside the
  stage's tmp context. `Pipeline.run()` skips the whole simulation if `sim_dir/.done` exists.
- **`Resources`**: `n_tasks`, `cpus_per_task`, `mem_per_cpu`, optional `walltime`; used by the CLI's SLURM
  submission helpers to build `sbatch` args. Optional `max_concurrent: int` caps how many jobs for that
  stage can be *running* at once across all simulations, independent of CPU/core headroom — enforced in
  `cli.py`'s `_submit_stage_slurm` via SLURM's `--dependency=singleton`, hashing each sim_dir into one of
  `max_concurrent` lanes that share a job name (so only one job per lane runs at a time). Useful for
  stages that shouldn't run too many in parallel regardless of free CPUs (e.g. a download stage hitting a
  rate-limited remote). Unset (`None`) by default — no cap, current `submit`-command behavior.
- Path validators: `TPath` (coerce to `Path`), `TPathExists` (must already exist), `TPathMkdir` (created on
  validation if missing). Always use these for new `Stage` fields.

More detail: `src/wrf_massive/README.md` (developer guide with a stage-implementation skeleton) and
`src/wrf_massive/stages/README.md` (one-line description of each of the four pipeline stages: forcing, wps, wrf,
postproc).

## The pipeline stages

1. **`forcing`** — retrieve/prepare forcing data for WPS. Two stages exist:
   - `PullCerraStage` (`stages/forcing/cerra.py`): reads a pre-built remote file listing (`remote_flist_path`,
     produced out-of-band via `find /path/to/cerra -type f > cerra_filelist.txt`), slices it to the simulation's
     `[begin_w_warmup, end]` window at the expected 3-hourly cadence, writes an `includes.txt`, and shells out to
     `rclone copy --include-from includes.txt ...` (`pull_cerra_db.tmpl.sh`) against a remote configured in the
     user's `rclone` config (`remote_path`, e.g. `tudelft:staff-umbrella/.../CERRA`). This assumes someone already
     mirrored/prepared GRIB files (CERRA `PRES`/`U10_V10`/`SFC`/`soil` file kinds) at a reachable rclone remote.
   - `PullCdsStage` (`stages/forcing/cds.py`): downloads GRIB directly from the Copernicus Climate Data Store via
     the official `cdsapi` client — no pre-mirrored data needed. Takes a `prefix` (must match Vtable/`fg_name`,
     e.g. `"CERRA"`/`"ERA5"`) and a list of `CdsRequestSpec` (one per CDS dataset/variable-set, e.g. pressure-level
     + single-level). Each request is downloaded as one CDS retrieve **per calendar month** (keeping the
     request's year/month/day cross-product waste-free across month boundaries), then each monthly file is
     **split into one GRIB file per valid time**, `<prefix>_<file_suffix>_<YYYYMMDD_HHMM>.grb`. The per-time split
     is required because WPS ungrib decodes a single valid time per input file (`ungrib/src/rd_grib2.F` reads the
     reference time once, from the first message, and applies it to the whole file), so a multi-time monthly file
     would otherwise collapse onto its first timestep. The split (`split_grib_by_reference_time`) is a pure
     byte-level copy that groups GRIB messages by their Section-1 reference time and handles both GRIB1 (ERA5) and
     GRIB2 (CERRA), independent of the product-definition template (CERRA uses the local template 4.214). The
     per-time files match `run_wps.sh`'s `find $FORCING_DIR -name '<prefix>*.grb'` glob (concatenated by
     `link_grib.csh`); re-runs skip already-produced per-time files and resume the split of any already-downloaded
     month. Per-request knobs: `product_type` (`"reanalysis"` for ERA5,
     `"analysis"` for CERRA), `use_area` (ERA5 supports lat/lon cropping to `Simulation.area`; CERRA's projected
     grid is pulled in full and cropped by WPS), and `extra_params` (e.g. CERRA's `data_type`/`level_type`).
     Area-cropped requests need `Simulation.area` (`BBox`), validated against the WRF domain (reprojected from the
     Lambert `namelist.wps` geogrid corners) via `stages/forcing/geo.py::validate_area_covers_domain`.
     Variable/level lists per `(source, level_type)` live in `stages/forcing/variables.py`. Note CERRA provides
     10m wind only as speed/direction (not the u/v components `Vtable.CERRA` needs), so source 10m u/v from ERA5.
   - An ARCO/Zarr-based fast path (ECMWF's Analysis-Ready Cloud-Optimized stores) was investigated but not
     implemented: it returns Zarr/xarray, not GRIB, so it would need a hand-written bridge into WPS's binary
     "intermediate format" — judged too risky/novel for the initial pass. Revisit if a lower-latency alternative to
     CDS's request queue is needed later.
2. **`wps`** (`stages/wps/wps.py`) — renders one `namelist.wps.<PREFIX>` per forcing source (currently `CERRA`,
   `ERA5`) from a single Jinja2 template, copies `Vtable.CERRA` + WPS run/setup scripts into the stage work dir, then
   `run_wps.sh` runs `geogrid.exe` once, `ungrib.exe` once per forcing source (linking `CERRA*.grb`/`ERA5*.grb` from
   `forcing_dir` via `link_grib.csh`, swapping in the right `Vtable`/`namelist.wps`), and finally `metgrid.exe`
   (`fg_name = 'ERA5','CERRA'`) to produce `met_em*` files.
3. **`wrf`** (`stages/wrf/wrf.py`) — renders `namelist.input`, links `met_em*` output, runs WRF.
4. **`postproc`** (`stages/postproc/`) — post-processes WRF output (e.g. `cn2.py`).

`stages/misc/` has small standalone stages: `MarkDone` (writes `.done`), plus `array.py`/`gc.py` helpers.
`StageArray` runs several substages inside one job (one SLURM allocation). It overrides `tmp_work_dir` only:
`tmp_work_root` set on the array applies to its *substages* (never to the array's own `work_dir`, which defaults to
`sim_dir`); a substage without an array-level root falls back to its own `tmp_work_root`, so substages can also be
relocated individually. A `contextlib.ExitStack` enters all of them up front so teardown happens only after *all*
substages have run — a later substage can then consume an earlier one's output while both are still on fast storage
(this is what the production `wrf` → `cn2` array relies on, and it is why the array cannot simply enter each
substage's context around its own `setup`/`run`). The stack also unwinds correctly when a substage raises. Roots are
passed as `tmp_work_dir(root=...)` keyword arguments rather than written onto the substages, because the same stage
instances are shared with other pipelines; per-substage behaviour (`tmp_teardown_globs`, `tmp_skip_teardown`) is
configured on the substages themselves. A substage that cannot be relocated is warned about and left in place.

## Which variables ungrib/WPS actually needs

Determined by the Vtable files linked in `run_wps.sh` — this is the authoritative list for any new forcing-data
source (CDS request `variable` lists, ARCO Zarr variable selection, etc. must cover these):

- **`Vtable.CERRA`** (`stages/wps/Vtable.CERRA`, ships in this repo) — pressure-level profile (level type `100`):
  `T`, `U`, `V`, `RH`, geopotential/height; near-surface single-level (level types `105`/`1`): `T2`, `U10`, `V10`,
  `RH2`, surface pressure, MSL pressure, land-sea mask, orography (`SOILHGT`), skin temperature, snow depth
  (physical + water-equivalent). This maps directly onto CERRA's mirrored file kinds: `PRES` = profile (pressure
  levels), `U10_V10` + `SFC` (+ `soil`, time-invariant/static) = single-level.
- **`Vtable.ERA-interim.pl`** (ships with the WPS submodule, `ungrib/Variable_Tables/`, used unmodified for ERA5) —
  pressure-level profile: `T`, `U`, `V`, `RH`/specific humidity, geopotential; single-level: `U10`, `V10`, `T2`,
  dewpoint/`RH2`, MSL/surface pressure, skin temp, sea ice, SST, snow (depth/density/water-equivalent), land-sea
  mask; plus 4 soil layers of soil temperature and moisture (`ST000007`...`ST100289`, `SM000007`...`SM100289`).

Net takeaway for a new download stage: split any new source into a **profile / pressure-level** request (temp,
u/v wind, humidity, geopotential across levels) and a **single-level** request (2m/10m fields, surface pressure,
MSL pressure, skin temp, snow, soil, land-sea mask, orography) — this split already exists in CERRA's file naming
(`PRES` vs `U10_V10`/`SFC`/`soil`) and lines up with how CDS/ARCO datasets are split (`reanalysis-*-pressure-levels`
vs `reanalysis-*-single-levels`).

## Running things

```bash
uv sync                                   # install/refresh env
cd workspaces/example
cp env_dev.yaml env.yaml                  # pick a machine profile
uv run cli.py init_sims simulations.py sim_test   # render sim_dir + namelists from a Simulation object
uv run cli.py run ./test_1                        # run all stages locally
uv run cli.py run --stages wps ./test_1           # run a subset of stages
uv run cli.py teardown --stages wps ./test_1      # force work dirs back from tmp storage (--all ignores globs)
uv run cli.py submit --jobfile slurm_hpc.sh ./test_1              # submit each stage as its own sbatch job
```

Tests: `uv run pytest`. See `tests/fixtures.py` for the `StageA`/`StageB` test-double pattern and
`simple_simulation` fixture used to exercise `Stage`/`Pipeline` behavior without real WRF/WPS binaries.

## Conventions to follow when extending this codebase

- New stages: subclass `Stage`, implement all four abstract methods, use `TPath`/`TPathExists`/`TPathMkdir` for path
  fields, keep it a `pydantic.BaseModel` (so it's YAML-serializable and CLI/pipeline-composable like existing
  stages), and prefer a `.done`-marker-friendly, idempotent `setup()`/`run()`.
- Keep `wrf_massive` project-agnostic; anything specific to a particular WRF study/domain goes in a
  `workspaces/<name>` directory, not in `src/wrf_massive`.
- Don't hand-roll subprocess calls; use `stages/utils.py::run_cmd_logged` (handles logging + non-zero exit) and
  `render_template`/`load_wps_wrf_namelist_tmpl` for Jinja2 rendering.
- Match the existing `forcing` stage naming/shape (`Pull<Source>Stage`) if adding new forcing-data stages, so they
  drop into `pipeline.py` the same way `PullCerraStage` does today.
