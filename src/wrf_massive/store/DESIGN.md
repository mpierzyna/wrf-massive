<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Store design

Where post-processed output goes. A postproc stage only reads and processes; its `store` writes:

- `NetCDFStore`: one netCDF file per input file in the stage's work dir (default).
- `ZarrStore`: one domain group of a zarr v3 store that holds many domains, laid out for ML training: parallel
  writes straight from post-processing, cheap random access to tiles of full columns, partial copies.

## Code

```
store/
├── base.py         PostProcStore (session, is_done, writer), Writer, StoreStage protocol
├── frames.py       frame times of input files, lead time indices, FrameTemplate
├── netcdf.py       NetCDFStore (writing and lazy frame-wise reading)
├── ingest.py       IngestStage: NetCDFStore output -> ZarrStore
└── zarr_store/
    ├── store.py    ZarrStore config (zarr imported only when used)
    ├── encoding.py Lossless, BitRound, FixedScaleOffset -> zarr codecs
    ├── layout.py   names, dims, attrs, locks
    ├── init.py     init-domain
    ├── write.py    session (preflight checks + sim lock), ZarrWriter, is_done
    ├── read.py     open_domain, hp_index, select_cells
    ├── verify.py   store verify
    └── pull.py     store pull
stages/parallel.py  FileParallelStage: per-file processing in a spawn process pool (PostProcStage, IngestStage)
```

Extra: `store = ["zarr>=3.4.2", "healpy>=1.18"]`. zarr is pinned to the version whose partial-shard writes and atomic
local writes were checked (below). `pull` needs the `rclone` binary.

## Store interface

`PostProcStore` models are configuration only (pydantic, `extra="forbid"`, chosen by `kind`; `TStore` is the union
used by `PostProcStage.store`). They are pickled to the workers; runtime state lives elsewhere:

- `session(stage, s, inputs)`: context in the parent around processing all inputs of a sim (checks, locks).
- `writer(stage, s, f)`: context in the worker yielding a `Writer` for input file `f`:
  `pending_frames(n)` (frames still to do) and `write(ds, i0)` (one or more consecutive frames starting at frame
  `i0` of `f`). Zarr commits per `write`; netCDF renames its temp file when the context is left.
- `is_done(stage, s, inputs)`.

A stage writing into a store implements `StoreStage`: `store`, `stored_vars(s)` (the variables it writes),
`store_template(s)` (one output frame and the file structure, to lay out a domain) and `input_static(f, names)`
(an input's static fields as the stage outputs them, for the grid check). Both `PostProcStage` and
`IngestStage` are `FileParallelStage`s: one input file per worker, started with `spawn` (zarr runs threads in the
parent, and forking a multi-threaded process can deadlock), so stages must be importable by module name and scripts
running them in parallel need an `if __name__ == "__main__":` guard.

## Zarr layout

```
<store>/
├── zarr.json                  root: schema_version, created_by; written once, never consolidated
├── alps/                      one self-contained group per domain, with consolidated metadata
│   ├── zarr.json              attrs: schema_version, interval_s, n_lead, shard_len, tile, warmup_h,
│   │                          static_vars, created_by, template, wrf_attrs (global attrs of the template)
│   ├── cn2/ th/ u_met/ ...    data: (sim, lead_time, bottom_top, y, x) or (sim, lead_time, y, x), float32
│   ├── XLAT XLONG HGT ...     static (y, x): lat/lon and `static_vars`, stored once
│   ├── hp_ipix                (y, x) uint64, HEALPix index
│   ├── lead_time              (lead_time,) seconds, decoded as timedelta
│   ├── time                   (sim, lead_time) valid time
│   ├── sim_name sim_begin     (sim,) identity, Simulation.begin
│   ├── sim_settings           (sim,) Simulation.settings as JSON
│   └── written                (sim, lead_time) bool commit mask
└── nl/ ...
```

Dims:

- `sim`: append-only position. A sim keeps its position forever (mask, shard keys, copies refer to it); new sims
  are appended in any order. `sim_name` (sim dir name, unique) is the identity; `sim_begin` need not be unique.
  Readers sort by `sim_begin`.
- `lead_time`: time since `Simulation.begin`. **The spin-up is excluded**: `lead_time = 0` is the first frame after
  it; frames before `begin` are refused. Number of lead times `(end − begin) / interval + 1`, the same for all sims
  of a domain; the interval is taken from the template.
- `bottom_top`: native WRF mass levels, index 0 lowest. No vertical interpolation; heights are data (`z` above sea
  level, `HGT`; height above ground is `z − HGT`). Staggered variables are refused (destagger in postproc).
- `y`, `x`: the mass grid (`south_north`, `west_east`), renamed because a domain can be rotated; optionally cropped
  by `crop` cells on each side (`PostProcStage.crop`, `IngestStage.crop`), e.g. WRF's lateral boundary zone
  (`spec_bdy_width`). The crop is applied after the postproc fns and recorded as `crop` in the output attrs.

Data arrays exist only for the variables a domain outputs (the union over domains is not padded), so an absent
variable raises `KeyError` instead of reading as NaN. Unwritten regions take no space and read as NaN.

### Chunks and shards

- Chunk (read and compression unit): `(1, 1[, all levels], ty, tx)`, `tile = (ty, tx)` default 64 × 64: one frame
  of full columns of a small tile (≈ 1.6 MB raw for 100 levels).
- Shard (one file): `(1, shard_len[, all levels], y, x)` with `shard_len` = frames per input file (6 for 6-hourly
  aux files), spatial extent rounded up to whole tiles (edge tiles padded with NaN). A shard never spans sims, so
  copies can be restricted by sim; it is written by exactly one worker.
- Chunk keys use the `.` separator: shards are flat files `c.<sim>.<shard>.0.0[.0]` in one dir per array, one inode
  each (the default `/` adds ≈ 3 directories per shard). Alps: 37 sims × 21 shards × 32 variables ≈ 25k files.

### HEALPix index

`hp_ipix(y, x)`: nested HEALPix cell of every grid point at order 23 (≈ 0.78 m, uint64), from `XLAT`/`XLONG`. Data
stays on the native grid; the index pairs it with other datasets (ERA5 ≈ order 8, CERRA ≈ 10, terrain/land use up
to 23) and selects cells. Coarser orders are exact bit shifts, `hp_ipix >> 2 * (23 − o)` (`read.hp_index`,
`read.select_cells`). At orders finer than the grid most cells hold no point; at coarse orders edge cells are partly
covered.

## Encoding

Data is float32 on disk; precision is reduced per variable by `ZarrStore.encoding` (default lossless):

| Encoding | Stored as | Error bound | For |
|---|---|---|---|
| `Lossless()` | float32 | none | small, static, sensitive fields |
| `BitRound(keepbits=n)` | float32, low bits zeroed | relative ≤ 2^-(n+1) | wide range (Cn², fluxes) |
| `FixedScaleOffset(offset, step)` | int32 `round((x − offset) / step)` | absolute ≤ step / 2 | differenced (θ) |

- Compressor: Blosc zstd(3), byte shuffle for lossless, bitshuffle after a lossy encoding (single-threaded in
  writers, which already run one process per file).
- `FixedScaleOffset` refuses NaN and values outside int32 (numcodecs would wrap silently). With `offset=300,
  step=1e-3` that is any θ from −2.1e6 to 2.1e6 K.
- The encoding is fixed at `init-domain` (codec chain plus an `encoding` attr per array); preflight refuses a stage
  whose config differs, as changing it means rewriting the array. One encoding may be shared by a project for the
  union of its domains' variables: entries for absent variables are reported, encodings of `static_vars` refused.
  `init-domain` warns if another domain encodes a variable differently.
- BitRound bounds the error relative to the value at every magnitude, i.e. it is a uniform quantization in log
  space (log10 error ≈ 2^-(n+1) / ln 10): store physical values and take log10 in the loader. A gradient is a small
  difference of large values: fields that are differenced need more bits than their values suggest, and θ (≈ 300 K)
  needs an absolute bound (see benchmarks).

The choice per variable belongs to the project (OTProf: `otprof_wrf/store.py`): 2D/static lossless, `th` 1 mK, `z`
and `p` BitRound 16, winds and humidity 12, turbulence fields 10 (−59 % vs. the zlib netCDF output on nl data).

## Writing

1. `wrf-massive store init-domain --stage NAME SIM_DIRS` (login node). Creates the store if needed and the domain
   group from a template (the stage's `store_template`: one processed frame of the first sim with aux files, or the
   first netCDF file for ingest), then appends the sims not yet in the domain by resizing `sim`. Metadata only, no
   space reserved; idempotent; consolidates the domain group (never the root). A per-domain `fcntl` lock next to the
   store (`<store>.<domain>.lock`) serializes runs for one domain; appending is safe while other sims are written
   (no existing chunk key changes, `zarr.json` is replaced atomically). The global `--debug` makes it a dry run.
2. `session` (before any frame is processed): the domain exists with the stage's variables, static vars, tile and
   encodings; the sim is initialized; every input file lies on the lead time grid after `begin`, within one shard,
   no shard twice; `XLAT`/`XLONG`/static fields of the input, as the stage outputs them (`input_static`, cropped),
   equal the stored ones, so another grid, domain or crop is refused. Then an exclusive `fcntl` lock in the stage's
   work dir (one job per sim) is held for the whole run.
3. `ZarrWriter` maps frame times to lead times, writes each variable into region `[sim, lead_time]` and sets
   `written` last: frames with `written = False` are redone after a crash (resume per frame). `is_done` = every frame
   of every input file is written.

Invariants: only `init-domain` writes metadata; every shard and every `written` chunk `(1, shard_len)` has exactly
one writer (one worker per input file, files aligned with shards, one job per sim).

Frame-wise writes into a multi-frame shard are read-modify-write: zarr reads the shard, re-encodes only the new
frame's chunks (the others are copied as compressed bytes) and writes the whole shard to `<key>.<uuid>.partial`,
renamed over the shard (atomic, so committed frames are never damaged). Cost: ≈ 3.5× write and 2.5× extra read I/O
for 6-frame shards, no extra compute; memory ≈ one shard of one variable. Ingest writes whole files, so each shard
is written once. Stray `.partial` files of killed writers are removed by `store verify --clean`.

`IngestStage(src=NetCDFStore(...), src_dir="4_postproc", store=ZarrStore(...))` copies existing netCDF output
(`NetCDFStore.root` reads `<root>/<sim name>/<src_dir>` of another workspace), one file per worker, one variable
loaded at a time; it also reads single-frame files of older versions that have a scalar `time`.

## Reading and copies

- `read.open_domain(store, domain)` opens the domain dir itself (`xr.open_zarr(<store>/<domain>,
  consolidated=True)`; xarray's `group=` looks for consolidated metadata at the root), sorted by `sim_begin`. Only
  frames with `written = True` hold data.
- A missing group raises, unlike with root consolidated metadata, which would list absent groups that then read as
  NaN. So a copy of the root `zarr.json` plus some domain dirs is a valid store.
- `wrf-massive store pull SRC DST --domain D [--vars] [--sims] [--rclone-config] [--transfers] [--dry-run]`
  (rclone; SRC `remote:path` or local): first the root and domain `zarr.json` and `sim_name` (to resolve sim names
  and globs to positions), then coordinates, static fields, the mask and the selected arrays' shards (filter file
  `- *.partial`, `+ <paths>`, `- **`). The copy's consolidated metadata is rebuilt to list only copied arrays, and
  `.wrf-massive-pull.json` records what it holds (merged over pulls); `open_domain` restricts a copy to that.
  `--dry-run` copies the small first pass and reports the size of the rest.
- `wrf-massive store verify (--stage NAME [SIM_DIRS] | --path STORE) [--domain] [--clean]`: per domain sims, frames
  written, incomplete sims, variables and size; a copy's manifest; stray `.partial` files.

## Not part of the store

Normalization statistics and train/validation splits (training side, splits by `sim_name`); vertical interpolation
and regridding (loader); spin-up frames. Adding a variable to an existing domain or changing an encoding needs a
rewrite of the array; all sims of a domain have the same lead times.

## Benchmarks (2026-11-20)

Encoding, nl `sim_2017-01-01`, file `2017-01-03_12` (6 frames, 26 variables, 796 MB raw; netCDF zlib: 602 MB):

| Codec chain (all variables) | Size |
|---|---|
| zstd(3) / Blosc zstd bitshuffle / Blosc zstd byte shuffle (lossless) | 625 / 503 / 492 MB |
| BitRound 16 / 12 / 10 / 8 + Blosc zstd bitshuffle | 330 / 231 / 184 / 140 MB |
| BitRound 16 / 12 / 10 / 8 + zstd(3) | 465 / 323 / 259 / 195 MB |
| OTProf encoding (above) | 245 MB |

2D fields are ≈ 1 % of the raw data. Gradients (same nl file, last row alps; lowest 40 levels; "zeroed" = a
non-zero gradient becomes exactly 0):

| `u`, `v`, `θ`, `z` encoded with | `dz` p99 err | `du/dz` zeroed | `S²` p99 err | `dθ/dz` zeroed | `dθ/dx` zeroed |
|---|---|---|---|---|---|
| BitRound 8 | 4 % | 3.6 % | 32 % | 57 % | 95 % |
| BitRound 12 | 0.25 % | 0.22 % | 2 % | 13 % | 47 % |
| BitRound 16 | 0.015 % | 0.01 % | 0.1 % | 1.4 % | 3.9 % |
| OTProf (θ 1 mK, z 16, u/v 12), alps | 0.05 % | 0.12 % | 3 % | 0.25 % | 0.39 % |

`θ`: BitRound 16 = 16.9 MB per nl file, `θ − 300` BitRound 14 = 21.3 MB, FixedScaleOffset 1 mK = 21.4 MB
(0.35 % / 1.0 % zeroed), lossless 32.5 MB: at equal size all encodings damage gradients alike. With 1 mK steps the
`N²` error is ≤ g/θ · 1 mK / 16.5 m ≈ 2e-6 s⁻² (stable boundary layers: 1e-4 to 1e-3 s⁻²).

Runs (local workstation): alps cn2 into a lossless store, 3 frames: 137 s and 2.6 GB peak (netCDF: 241 s, 2.4 GB),
output identical; init-domain from an alps aux file: ≈ 50 s, ≈ 5.6 GB. Ingest of a full nl sim (21 files) with the
OTProf encoding: 39 s with 8 workers, 12.3 → 5.11 GB, 625 inodes, ≈ 1.3 GB per worker; all errors within their bounds.
