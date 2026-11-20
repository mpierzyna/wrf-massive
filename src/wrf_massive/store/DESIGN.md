<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Store design (draft for review)

One zarr v3 store with post-processed output of all domains, written in parallel by postproc jobs and read by
ML training. Decisions are marked **[accepted]**, **[proposed]** (ready to review), **[open]** (needs input) or
**[tbd]** (deferred).

## Layout

```
otprof.zarr/
├── zarr.json                  # root: minimal attrs (schema version), written once; NOT consolidated
├── alps/                      # one self-contained group per domain, own consolidated metadata
│   ├── zarr.json
│   ├── cn2/  tk/  u/ ...      # data arrays (sim, lead_time, bottom_top, y, x) or (sim, lead_time, y, x)
│   ├── time                   # (sim, lead_time) datetime64, valid time
│   ├── sim_name               # (sim,) str, sim dir name, unique: the sim's identity
│   ├── sim_begin              # (sim,) datetime64, `Simulation.begin` (end of spin-up), not unique
│   ├── sim_settings           # (sim,) str, `Simulation.settings` as JSON
│   ├── XLAT  XLONG  HGT  LANDMASK ...  # static fields (y, x), stored once, not per frame
│   ├── hp_ipix                # (y, x) uint64 HEALPix index (nested, order 23), see below
│   └── written                # (sim, lead_time) bool, commit mask
└── baltics/ ...
```

- **[accepted] Each domain group is self-contained.** Grids differ, so arrays cannot be shared across domains
  anyway. A group carries everything needed to use it alone: schema version, encoding, coords, static fields,
  sim metadata. Copying `zarr.json` + `alps/` gives a valid one-domain store.
- **[accepted] No consolidated metadata at the root.** It would list groups that are absent in a partial copy, and
  reading those silently returns the fill value (verified with zarr 3.4.2). Without it, a missing group raises
  `KeyError` / `GroupNotFoundError`. Consolidate per domain group instead.
- **[accepted] One array per variable** (per-variable encoding, pulling only some variables).
- **[accepted] `sim` is the leading dim and shards never span sims**, so shard keys are
  `<domain>/<var>/c.<sim>.<shard>.0.0.0` and a copy can be restricted to some sims or some variables by path
  filters.

## Dims, chunks, shards

- **[accepted] `sim`: stable, append-only position**, not an ordering. Sim *i* is always the same sim (the
  `written` mask, shard keys `c.<sim>.…`, pull filters and manifests refer to positions); new sims are appended
  in any order, never inserted. `read.py` sorts by `sim_begin` (lazy orthogonal selection, no copy) by default;
  physical order does not matter for reading since no read spans sims.
- **[accepted] `sim_name` is the identity** (sim dir name, e.g. `sim_2017-01-05`); splits and manifests refer to
  it, not to positions. `sim_begin` need not be unique (e.g. same start date with different physics), so
  `Simulation.settings` are stored per sim as a JSON string (`sim_settings`); settings keys can then differ
  between sims without new arrays.
- `lead_time`: duration since `Simulation.begin`, alps: 0..120 h (121 frames, hourly). **SPIN-UP IS EXCLUDED**: the
  store holds only post-processed, usable data, so `lead_time=0` is the first frame *after* the spin-up (12 h after WRF
  start). Writes reject any frame before `Simulation.begin` (postproc's `discard_warmup` drops whole files only).
  `warmup_h` is kept as a group attr for provenance only.
- `bottom_top`: native WRF mass levels (100), named as in WRF so the order is clear (index 0 = lowest
  level). Not `z`, which is the height variable. Staggered dims are rejected at write time (destagger in postproc).
- **[accepted] `y`, `x`**: mass grid (WRF `south_north`, `west_east`). Not the WRF names: domains can be rotated,
  so the grid axes need not point south–north / west–east.
- **[accepted] Shard = one aux file = `(1, 6, bottom_top, y, x)`.** One postproc worker owns exactly one shard per
  variable, so parallel writers never touch the same file. 21 shards per sim (the last holds only frame 120).
  Alps: 37 sims × 21 shards × 32 time-dependent vars ≈ 25k files; all domains ≈ 130k.
- **[accepted] Chunk key separator `.`** (`chunk_key_encoding={"name": "default", "separator": "."}`): shards are
  flat files in one directory per array. The default `/` nests a directory per key index, ≈ 3 extra directories
  (inodes) per shard (verified with zarr 3.4.2).
- **[accepted] Inner chunks: all levels, small horizontal tiles, one frame**: `(1, 1, bottom_top, ty, tx)`. Proposed
  `ty = tx = 64` (≈ 1.6 MB raw for 100 levels); final tile size by a read benchmark (32 / 64 / 128).

## Preallocation and parallel writes

- **[accepted] `wrf-massive store init-domain --stage NAME SIM_DIRS...`** (dashes like `init-sims`). Help text: "Add the
  domain of the given postproc stage and the given sims to the store. Creates a new store if this is the first domain,
  appends the domain to an existing store otherwise, and appends sims not yet in the domain." Takes sim dirs like
  `run`/`submit`; it never scans the workspace. `--stage` is mandatory and must name a stage in `pipeline.py` that
  writes to a `ZarrStore` (`PostProcStage.store` or `IngestStage.store`; no default: wrf-massive stays
  project-agnostic). Store path, domain group and the variables to allocate come from that stage, so they are
  configured once.
  - **Creating a domain needs one sim with output (template).** The stage's variable list gives names, but
    not dims, grid, levels or attrs, and postproc fns (e.g. Cn2) may return anything. If the first given sim has
    netCDF postproc output, its first file is the template (needed for nl/canaries, whose wrfout is deleted).
    Otherwise creating a domain runs the stage's `_process_frame` on frame 0 of the first given sim that has aux
    files: this yields exact dims, dtype and attrs of every stored variable, the static fields (`XLAT`,
    `XLONG`, `HGT`, `LANDMASK`, `hp_ipix`) and the frame interval, and doubles as a test that postproc works.
    Costs one frame (alps: ≈ 1–1.5 min, ≈ 3.5 GB). No given sim with netCDF output or aux files and no domain
    group yet → error. Deriving shapes from
    the namelist instead would need declared output dims for every variable and postproc fn.
  - **Appending needs no WRF output**: only `simulation.yaml` (`sim_name`, `begin`/`end`, settings), so all sims
    of a domain can be initialized right after the first one finished WRF. Sims already present are skipped
    (idempotent). Raises if `end - begin` differs from the domain's lead time range. The grid check against
    the static fields moves to the preflight check at postproc start (`session`), where the aux files exist.
  - Metadata only, no space reserved: an empty alps group with 3 full-size 3D arrays (0.4 TB logical each) is
    4.7 KB on disk; data appears only as frames are written (verified with zarr 3.4.2). Only `init-domain`
    touches metadata; workers only write chunk data.
  - Safe while postproc of already initialized sims writes: appending changes no existing chunk key and zarr
    writes `zarr.json` atomically (tmp + rename). Two `init-domain` runs for the same domain must not overlap;
    each takes an exclusive lock (`fcntl`) on a per-domain lock file next to the store (`<store>.<domain>.lock`,
    not inside it, so copies stay clean).
  - **Domains are added the same way**: `init-domain` on an existing store creates the missing domain group. The
    root is created by the first `init-domain` of any domain and never changed afterwards (no consolidated root
    metadata, and a zarr v3 group's `zarr.json` lists no children), so a new domain touches neither the other
    groups nor their running writers; runs for different domains may overlap. Raises if the store's schema
    version differs from the code's. Each domain's workspace points its `ZarrStore.path` at the same store.
- **[accepted] Workers write frame by frame, region `[sim, t]`**, straight from postproc. With
  `write_empty_chunks=False` and `fill_value=NaN` (floats), unwritten regions cost no disk space and read as NaN.
- A frame write covers only part of its 6-frame shard, so zarr (3.4.2, checked in the source) reads the whole
  shard, re-encodes only the new frame's chunks (the others are copied as compressed bytes, not recompressed) and
  writes the whole shard again. Cost: I/O, not compute: 21 frame writes + 15 frame reads per shard instead of 6
  writes (alps: ≈ 400 GB written + 290 GB read per sim instead of 116 GB; ≈ 60 MB/s per worker), memory ≈ one
  shard of one variable (≤ 0.5 GB for an alps 3D variable).
- The local store writes `<key>.<uuid>.partial` and renames it over the shard (atomic on the same file system), so
  a crash never damages frames already in the shard. The read-modify-write has no locking: **one writer per
  shard is mandatory** (one worker per aux file guarantees it).
- **[accepted] Commit mask `written`** (chunks `(1, shard length)`, owned by the same worker as the shards) is set after
  each frame, so a crashed job leaves `False` and a restart resumes per frame. Readers and `store verify` trust only
  `written=True`; `store verify --clean` removes stray `*.partial` files left by a hard kill.
- Appending (resize of `sim`, new per-sim coords, re-consolidation) leaves existing shards untouched. All sims
  must have the same 121 lead times.
- Shard shape must be a multiple of the chunk shape: alps 524×419 with 64×64 tiles → shards of 576×448 (9×7
  tiles). Edge tiles are stored padded with NaN (cheap after compression); tiles fully outside the grid do not
  exist.

### Preflight check at postproc start (`PostProcStore.session`)

**[accepted]** Before any frame is processed (fail fast, nothing written), the zarr backend checks:

1. The domain group exists and matches the stage: every variable the stage writes has an array with the
   expected dims (`(sim, lead_time, bottom_top, y, x)` or `(sim, lead_time, y, x)`; anything else, e.g. a
   staggered dim, raises with a hint to destagger), dtype, chunks and shards.
2. `sim_name` is present exactly once → its position. Otherwise: "sim not initialized, run
   `wrf-massive store init-domain ...`".
3. The sim's input files fit (aux files for postproc, netCDF files for ingest): grid shape, `XLAT`/`XLONG` and
   `static_vars` equal the stored static fields, every file starts at a shard boundary and holds ≤ shard length
   frames, all lead times within the domain's range.
4. No other job writes this sim: an exclusive POSIX lock (`fcntl`, works across nodes on GPFS) on a lock file in
   the stage's work dir, held for the whole run. Outside the store, so copies stay clean. Guards against
   two jobs for the same sim (one writer per shard).

The backend never writes metadata. `is_done` of the stage is "all `written[sim, :]` are True"; on restart frames
with `written=True` are skipped.

## Store abstraction in postproc

**[accepted]** Everything related to writing moves out of `PostProcStage` into a `store` property. netCDF and zarr
are two implementations of the same interface; the stage keeps reading and processing.

| `PostProcStage` (reading, processing) | `store` (writing) |
|---|---|
| `extract_vars`, `postproc_fns`, `myoutfields_path`, `iofields_stream` | `compression`, `file_suffix` → `NetCDFStore` |
| `discard_warmup`, `get_inputs` (input selection) | `_encoding`, `_append_frame`, tmp file + rename |
| `run_parallel`, `n_workers`, `_process_frame` | `is_done`, resume logic |
| `work_dir` (logs, `.gitignore`, zarr sim lock) | output location: netCDF `work_dir`, zarr `path` + `domain` |

- **Typed pydantic models, discriminated by `kind`**, not a free `store_config` dict, so typos raise when the
  pipeline is loaded. In YAML: `store: {kind: zarr, path: ..., domain: ...}`.
  - `NetCDFStore(compression=True, file_suffix="proc", root=None)`: today's behaviour, the default. Files are in
    the stage's work dir, or in `<root>/<sim_name>/<work_dir>` if `root` is set (reading other workspaces).
    Also readable: `outputs(stage, s, work_dir)` and `read_frames(f, i0, n)` (lazy), used by `IngestStage`.
  - `ZarrStore(path, domain, tile=(64, 64), static_vars=[], encoding={})`: `static_vars` (e.g. `HGT`,
    `LANDMASK`) are written once at `init-domain`, compared at preflight and dropped from every frame by the
    writer; default empty so wrf-massive stays project-agnostic.
- **Config vs runtime state**: store models are configuration only (pickled to pool workers, YAML-serializable).
  Open files and locks live in short-lived writers:

  ```python
  class PostProcStore(pydantic.BaseModel, abc.ABC):
      def session(self, stage, s, inputs) -> ContextManager  # parent, around the pool (zarr: checks + sim lock)
      def is_done(self, s, inputs) -> bool                   # PostProcStage.is_done delegates here
      def writer(self, stage, s, f) -> ContextManager[Writer]  # one per input file, in the worker

  class Writer:
      def pending_frames(self, n: int) -> list[int]          # netCDF: all or none; zarr: frames with written=False
      def write(self, ds, i0) -> None                        # ds: n >= 1 frames, starting at frame i0 of the file
  ```

- The stage's run loop is store-agnostic; postproc stays frame-wise (memory ≈ one frame), no netCDF intermediate:

  ```python
  with self.store.writer(self, s, f) as w:
      for i in w.pending_frames(n):
          w.write(self._process_frame(f, i), i)
  ```

- `write(ds, i0)` with several frames covers full shards: zarr then writes each shard once (1× I/O, no
  read-modify-write). The zarr writer loads one variable at a time, so memory stays ≈ one shard of one variable.
- `lead_time = frame time − Simulation.begin`; negative values raise (spin-up).
- **Ingest** (`IngestStage`, see CLI) is generic: frames read from a `NetCDFStore`'s output are written in whole
  files (6 frames) into a `ZarrStore`. Existing output (checked 2026-11-20), all float32, 21 files per sim,
  spin-up excluded:

  | Domain | Sims | Size | Location |
  |---|---|---|---|
  | alps | 37 | 4.2 TB | `prjs0951/otprof-wrf/domains/alps/sim_*/4_postproc` |
  | nl | 73 | 836 GB | `/projects/prjs1785/otprof/data/nl/wrf/sim_*/4_postproc` |
  | canaries | 12 | 122 GB | `/projects/prjs1785/otprof/data/canaries/wrf/sim_*/4_postproc` |

  Quirks ingest handles (older wrf-massive output): the last, 1-frame file of nl/canaries has a scalar `time`
  (no `time` dim; expand it); stale `stagger="Z"` attr on destaggered `EL_PBL` (dims are authoritative, the attr
  is dropped). nl/canaries variables: 22 shared with alps plus `mdbz`, `rwp`, `swp`, `iwv` (union).
- The writer renames WRF's horizontal dims to `y`/`x` (`south_north`/`west_east` in all postproc output).
- Generic sizes in code: shard length = frames per aux file (from the template), number of lead times =
  `(end − begin) / interval + 1`, `lead_time` stored as a duration. The numbers in this document (6, 121, hourly)
  are the alps example.
- Breaking change for workspaces (`compression=`/`file_suffix=` on the stage → `store=NetCDFStore(...)`); released
  as a minor version, workspaces are migrated.

## Partial copies (subset of domains / sims / variables)

- Missing chunks read as NaN without error, and the `written` mask of a copy still describes the source.
- **[accepted] `store pull --domain alps [--vars ...] [--sims ...] [--rclone-config PATH] SRC DST`** wraps `rclone copy`
  (any rclone remote: sftp to Snellius, S3, local, ...). It copies metadata and coords first, maps `--sims` names to
  positions via `sim_name`, then turns the filters into rclone `--include` rules; `--rclone-config` is passed on as
  `rclone --config` (default: rclone's own config). It excludes `*.partial` files and writes a local manifest of what
  was copied. The loader checks requested `(domain, var, sim)` against the manifest and raises instead of returning NaN.
- Reading remotely (fsspec over ssh) works for inspection, not for training.

## Variables and vertical levels

- **[accepted] Variable set: union of all domains.** The store schema lists the union; a domain group only
  contains the arrays the domain actually has (nl/canaries: no QSQ/COV/WRF-Solar fields, but `mdbz`, `rwp`,
  `swp`, `iwv`), so asking
  for an absent variable raises `KeyError` instead of returning NaN. Static fields and units/description attrs
  are taken from the wrf-python output.
- **[accepted] Native model levels.** No vertical interpolation in the store. Heights are data, not coordinates:
  `z` (height ASL, 4D, already in the postproc output) and `HGT` (static) are stored like every other variable;
  height above ground is `z - HGT`. Any interpolation to fixed heights happens in the loader.

## HEALPix index (alignment with other datasets)

- **[accepted] Data stays on the native grid**; HEALPix is only an index for pairing with other datasets (ERA5,
  CERRA, high-res terrain/land use) and for selecting cells.
- **[accepted] One static field `hp_ipix(y, x)`, uint64, nested scheme, order 23** (≈ 0.78 m cells, the first
  order below 1 m; ≈ 50 bits). Computed once per domain from `XLAT`/`XLONG` (cast to float64), e.g.
  `healpy.ang2pix(2**23, xlong, xlat, nest=True, lonlat=True)`. Attrs: `order=23`, `indexing_scheme="nested"`,
  lat/lon source. Same order for every domain.
- Every coarser order is exact and needs no extra storage: `ipix_o = hp_ipix >> 2 * (23 - o)` (a point inside a
  cell is inside its parent). Pairing with another dataset works through any order both share, e.g. ERA5 at
  order 8 (≈ 25 km), CERRA at 10 (≈ 6.4 km).
- It is an index, not a grid: at orders finer than the 2 km spacing most cells contain no point; at coarse orders
  edge cells are only partly covered (coverage ≈ points in cell × native cell area / HEALPix cell area, computed
  in the loader, not stored).
- `read.py` provides `hp_index(order)` and `select_cells(cells, order)`, so nobody writes the shifts by hand.

## Encoding

**Summary.** float32 on disk; precision is reduced per variable by one of three encodings. 2D and static fields
stay lossless, 3D fields get a precision matched to how they are used. Suggested OTProf setting: 245 MB per nl
file instead of 602 MB netCDF (−59 %), nl ≈ 340 GB instead of 836 GB.

### Mechanism (wrf-massive)

- **[accepted] float32 on disk** for every variable (as postproc). float16 underflows Cn² (min normal 6e-5) and
  overflows `p` in Pa (max 65504); bfloat16 has only 8 significant bits (2 K steps at 300 K) and is neither a
  numpy nor a core zarr v3 dtype. Training can still cast on load.
- **[accepted] `ZarrStore.encoding: dict[str, Encoding] = {}`**, small pydantic models (validated,
  YAML-serializable), mapped to `zarr.codecs.numcodecs`:

  | Encoding | Error bound | Use for |
  |---|---|---|
  | `Lossless()` (default for unlisted variables) | none | small or sensitive fields |
  | `BitRound(keepbits=n)` | relative, ≤ 2^-(n+1) of the value | fields spanning orders of magnitude |
  | `FixedScaleOffset(offset, step)` | absolute, ≤ step/2 | fields whose differences matter (`θ`) |

  `FixedScaleOffset` stores `(x − offset) / step` as int32; the writer checks the range and raises on overflow
  (numcodecs wraps silently). `step` instead of numcodecs' `scale = 1/step`, so the config reads "1 mK".
- **[accepted] Compressor follows automatically**: Blosc zstd(3) with byte shuffle for lossless, bitshuffle after
  a lossy encoding. Users choose precision, not codec chains.
- **[accepted] Fixed at `init-domain`**: written into each array's codec chain plus a readable `encoding` attr.
  Preflight raises if the stage's config differs from the stored codecs (changing it means rewriting the array);
  `init-domain` warns if another domain encodes the same variable differently.
- A project may share one `encoding` for the union of its domains' variables: `init-domain` reports entries
  for variables a domain does not output (e.g. `QSQ`/`COV` for nl) instead of failing; encodings for
  `static_vars` raise (they are always lossless).
- Readers other than zarr-python (tensorstore, zarrs) may not support numcodecs codecs; check before choosing a
  non-Python loader.

### OTProf choices (otprof-wrf)

**[accepted]** One shared `OTPROF_ENCODING` (union of variables) next to `OTProfCn2PostProcStage`, used by every
domain (`ZarrStore(path=env["store_path"], domain=..., encoding=OTPROF_ENCODING, static_vars=["HGT",
"LANDMASK"])`), so all domains encode alike. Four rules:

1. **2D and static fields: lossless.** They are 1.2 % of the raw data (nl: 9 of 796 MB per file); no error is
   worth that saving.
2. **Fields the loader differences or differentiates: absolute or near-absolute precision.** `th`:
   `FixedScaleOffset(offset=300, step=1e-3)` **[accepted]**; `z`, `p`: `BitRound(16)`.
3. **Winds and humidity: `BitRound(12)`** (0.012 %; shear error p99 2 %).
4. **Turbulence fields: `BitRound(10)`** (0.05 %; for Cn² 2e-4 decades in log space). BitRound keeps the
   exponent exactly and bounds the error relative to the value, so it is a uniform quantization in log space:
   log10 error ≈ 2^-(n+1) / ln 10 at every magnitude (n=8: 8.5e-4 decades, n=6: 3.4e-3). Converting to log10
   first gains nothing (BitRound on log values gives a non-uniform error of |log10 x| · 2^-(n+1), FixedScaleOffset
   on log values equals BitRound on physical values) and breaks on zeros (1.2 % of Cn²) and negative `COV`.
   The store stays physical; log10 is the loader's choice. n=10 **[accepted]**.

| Variable (3D) | Encoding | Error | nl MB/file (lossless) |
|---|---|---|---|
| `th` | FixedScaleOffset 1 mK | ≤ 0.5 mK | 21.4 (32.5) |
| `z` | BitRound 16 | ≤ 7.6e-6 rel (≤ 0.03 m at 3 km) | 18.7 (34.3) |
| `p` | BitRound 16 | ≤ 7.6e-6 rel (≤ 0.8 Pa at 1000 hPa) | 13.6 (29.4) |
| `u_met`, `v_met`, `w` | BitRound 12 | ≤ 1.2e-4 rel | 19.8, 21.0, 33.6 (42.3, 43.1, 54.9) |
| `rh` | BitRound 12 | ≤ 1.2e-4 rel | 18.7 (39.9) |
| `cn2` `ct2` `QKE` `TSQ` `EL_PBL` `QSQ` `COV` | BitRound 10 | ≤ 4.9e-4 rel | 22.5, 22.6, 9.0, 22.3, 15.8 (≈ 2.2×) |
| all 2D, static | lossless | 0 | 6.0 total |

### Evidence (benchmark 2026-11-20)

nl `sim_2017-01-01`, file `2017-01-03_12` (6 frames, 26 variables, 796 MB raw), chunks `(1, 100, 64, 64)`,
6-frame shards, real zarr writes. Scripts: `bench_enc.py`, `bench_grad.py`, `bench_th.py` (not in the repo).

- **Compressors (lossless):** zstd(3) 625 MB, Blosc zstd + bitshuffle 503 MB, Blosc zstd + byte shuffle 492 MB;
  current netCDF (zlib) 602 MB. Turbulence fields, fluxes and `w` barely compress losslessly.
- **BitRound, all variables:** n = 16 / 12 / 10 / 8 → 330 / 231 / 184 / 140 MB with bitshuffle (≈ 30 % smaller than
  with plain zstd). The relative error is exactly 2^-(n+1) for every variable. `rwp`/`swp` contain subnormal
  noise (down to 1e-44) that BitRound flushes to 0.
- **Gradients** (lowest 40 levels, `u`, `v`, `θ`, `z` all BitRounded with the same n; "zeroed" = a non-zero
  gradient becomes exactly 0, the only sign change seen):

  | n | `dz` p99 rel. err | `du/dz` zeroed | `S²` p99 rel. err | `dθ/dz` (N², Ri) zeroed | `dθ/dx` zeroed |
  |---|---|---|---|---|---|
  | 8 | 4 % | 3.6 % | 32 % | 57 % | 95 % |
  | 12 | 0.25 % | 0.22 % | 2 % | 13 % | 47 % |
  | 16 | 0.015 % | 0.01 % | 0.1 % | 1.4 % | 3.9 % |

  A gradient is a small difference of large values; relative rounding hurts `θ` (≈ 300 K, step 2^(8−n) K) whose
  level-to-level `Δθ` has median 0.3 K, p10 0.02 K, and neighbour `Δθ` median 0.03 K.
- **`θ` encodings:**

  | `th` encoding | Size | max abs err | `dθ/dz` zeroed | `dθ/dx` zeroed |
  |---|---|---|---|---|
  | lossless | 32.5 MB | 0 | 0 | 0 |
  | BitRound 16 | 16.9 MB | 3.9 mK | 1.4 % | 3.9 % |
  | `θ − 300`, BitRound 12 | 16.9 MB | 31 mK | 1.4 % | 3.2 % |
  | `θ − 300`, BitRound 14 | 21.3 MB | 7.8 mK | 0.35 % | 0.8 % |
  | FixedScaleOffset 1 mK | 21.4 MB | 0.5 mK | 0.35 % | 1.0 % |
  | FixedScaleOffset 0.1 mK | 28.7 MB | 0.05 mK | 0.03 % | 0.07 % |

  At equal size all encodings damage gradients alike (the offset only moves ≈ 4 bits of precision to the lower
  troposphere). Remaining zeroed gradients are near-neutral: with 1 mK steps the `N²` error is
  ≤ g/θ · 1 mK / 16.5 m ≈ 2e-6 s⁻² (stable boundary layers: 1e-4 to 1e-3 s⁻²).

- **Alps cross-check** (`sim_2017-01-05`, local file `2017-01-04_18` with 3 frames, 18–20 UTC, inside the spin-up;
  34 variables, 524×419×100, 3.74 GB raw; script `bench_alps.py`):
  - Size: netCDF 3051 MB, lossless 2620 MB, accepted encoding **1404 MB** (−54 % vs netCDF, −46 % vs
    lossless; nl: −59 %). 2D fields are 1.4 % of the raw data.
  - Error bounds hold for every variable: BitRound relative error exactly 2^-(n+1) (incl. negative `COV`),
    `th` ≤ 0.52 mK, `z` ≤ 0.13 m at 29.7 km.
  - Gradients with the accepted encoding (lowest 40 levels), all columns / columns with `HGT` > 1500 m (7.7 %):
    `dz` p99 0.05 % / 0.11 %, `du/dz` zeroed 0.12 % / 0.16 %, `S²` p99 3.0 % / 2.9 %, `dθ/dz` zeroed 0.25 % /
    0.08 %, `dθ/dx` zeroed 0.39 % / 0.07 %. High terrain is not worse for `θ` (larger `Δθ`); `dz` errors double
    there (larger `z`) but stay ≈ 0.1 %.
  - Subnormals only in 2D fields (`RAINNC`, `WP_TOT_SUM`, `LWP_TOT`, `IWP_TOT`), which stay lossless.

## Normalization stats and splits

- **[accepted] Not part of the store.** Normalization stats are computed later on the training side; splits are
  defined by `sim_name` in the training config.

## CLI

**[accepted]** A `store` group next to `plot`. Commands that need a stage load the workspace's `pipeline.py` (run
from inside the workspace, like `run`/`submit`); `verify` on a path and `pull` work without a workspace.

```
$ wrf-massive store --help
Usage: wrf-massive store [OPTIONS] COMMAND [ARGS]...

  Zarr store of post-processed output.

Commands:
  init-domain  Add a domain and/or sims to a store (creates the store if needed).
  verify       Report what a store (or a local copy) contains; optionally clean up.
  pull         Copy (part of) a store with rclone.
```

### `init-domain`

```
Usage: wrf-massive store init-domain [OPTIONS] SIM_DIRS...

  Add the domain of the given postproc stage and the given sims to the store. Creates a new store if this is
  the first domain, appends the domain to an existing store otherwise, and appends sims not yet in the domain.

Options:
  --stage NAME  Stage in pipeline.py that writes to a ZarrStore.  [required]
```

`wrf-massive --debug store init-domain ...` only prints what would be done (the global `--debug`, as for `submit`).

```
domains/alps$ wrf-massive store init-domain --stage cn2 sim_*
INFO  Store /gpfs/work4/0/prjs0951/otprof.zarr does not exist -> creating (schema 1)
INFO  Domain 'alps' does not exist -> creating from template sim_2017-01-05/4_postproc/wrfout_aux_..._cn2.nc
INFO  -> 32 variables (14 3D, 18 2D), 2 static, grid 419 x 524, 100 levels, 121 lead times, 6 per shard
INFO  Appending 36 sims (1 already present: sim_2017-01-05)
```

### Postproc writes via the normal `run` / `submit`

No new command: the stage's `store` decides where frames go.

```
domains/alps$ wrf-massive submit --stages cn2 sim_*        # each job: preflight, then frames -> store
```

### Ingest existing netCDF output: a stage, not a command

**[accepted]** Ingest is hours of I/O per domain (alps 4.2 TB), needs
SLURM resources, runs per sim and resumes per file — exactly a stage. So `IngestStage(src=NetCDFStore(...),
src_dir="4_postproc", store=ZarrStore(...))` in `pipeline.py`, run with `run`/`submit`; `is_done` = all frames
`written`. The destination is called `store` like `PostProcStage.store` (`init-domain`, checks and writer are
shared). `NetCDFStore(root=DIR)` reads `<DIR>/<sim_name>/<src_dir>` instead of the sim's own dir, which covers
nl/canaries in the old prjs1785 workspaces (they stay read-only; no `--src-root` option, no symlinks). Each worker
writes whole files, loading one variable at a time (`read_frames` is lazy).

Measured (2026-11-20, local, nl `sim_2017-01-01`, 21 files, accepted encoding): 39 s with 8 workers, 12.3 GB
netCDF → 5.11 GB in the store (−58 %), 625 inodes, 1.3 GB peak per worker; all errors within their bounds.

```python
# domains/nl/pipeline.py (snellius)
_ingest = IngestStage(
    work_dir="5_ingest",  # logs + sim lock only
    src=NetCDFStore(root="/projects/prjs1785/otprof/data/nl/wrf", file_suffix="cn2"),
    store=ZarrStore(path="/gpfs/work4/0/prjs0951/otprof.zarr", domain="nl", static_vars=["HGT", "LANDMASK"]),
    run_parallel=True,
    n_workers=8,
    resources=Resources(n_tasks=1, cpus_per_task=24, mem_per_cpu="1750M", walltime=...),
)
```

```
domains/nl$ wrf-massive store init-domain --stage ingest sim_*   # template from the netCDF output
domains/nl$ wrf-massive submit --stages ingest sim_*
```

### `verify`

```
Usage: wrf-massive store verify [OPTIONS] [SIM_DIRS]...

  Report what a store or a local copy contains. With --stage: the stage's store, limited to SIM_DIRS if given.
  With --path: any store or copy, no workspace needed.

Options:
  --stage NAME   Stage whose ZarrStore to check.
  --path PATH    Store (or copy) to check instead.
  --domain NAME  Only this domain (repeatable).
  --clean        Remove stray `*.partial` files (only while nothing writes).
```

Example output (sizes illustrative):

```
$ wrf-massive store verify --path otprof.zarr
otprof.zarr  (schema 1)
  alps      37 sims   4477/4477 frames written   32 vars   2.9 TB on disk
  baltics   37 sims   1210/4477 frames written   32 vars   0.8 TB   incomplete: 27 sims
  nl        73 sims   8833/8833 frames written   24 vars   0.6 TB
  local copy (manifest): alps only, vars cn2 z HGT, sims 10/37
  stray .partial files: 0
```

### `pull`

```
Usage: wrf-massive store pull [OPTIONS] SRC DST

  Copy (part of) a store with rclone. SRC/DST are rclone paths (remote:path or local). Metadata, coordinates,
  static fields and the `written` mask are always copied; data arrays are filtered. Writes a manifest of what
  was copied into DST, which the readers check.

Options:
  --domain NAME          Domain to copy (repeatable).  [required]
  --vars VAR,...         Only these variables (default: all).
  --sims NAME,...        Only these sims, by sim_name; globs allowed (default: all).
  --rclone-config PATH   rclone config file (passed as `rclone --config`; default: rclone's own).
  --transfers N          Parallel transfers (default: 8).
  --dry-run              Show what would be copied.
```

```
laptop$ wrf-massive store pull snellius:/gpfs/work4/0/prjs0951/otprof.zarr ~/data/otprof.zarr \
          --domain alps --vars cn2,z,u_met,v_met --sims 'sim_2017-*' --rclone-config ~/.config/rclone/hpc.conf
```

## Code

```
wrf_massive/store/
├── base.py     # PostProcStore (session, is_done, writer), Writer
├── frames.py   # frame times of input files, lead time indices, FrameTemplate (no zarr needed)
├── netcdf.py   # NetCDFStore (today's netCDF writing, moved from stages/postproc/base.py)
├── zarr_store/ # not `zarr/`, to never shadow the zarr package; zarr imported only when used
│   ├── encoding.py # Lossless, BitRound, FixedScaleOffset (pydantic) -> zarr codecs
│   ├── store.py    # ZarrStore config, delegates to the modules below
│   ├── layout.py   # names, dims, attrs, locks shared by init/write/read
│   ├── init.py     # init-domain: create store/domain group, append sims (preallocation), lock
│   ├── write.py    # session (preflight checks + sim lock), ZarrWriter, is_done
│   ├── read.py     # open_domain (sorted, restricted to a copy's manifest), hp_index, select_cells
│   ├── verify.py   # store verify
│   └── pull.py     # rclone wrapper + manifest
├── ingest.py   # IngestStage (NetCDFStore → ZarrStore)
│   (zarr_store/verify.py: `store verify`; zarr_store/read.py: open_domain, hp_index, select_cells)
└── (CLI)       # wrf-massive store init-domain | verify | pull
```

- zarr and healpy as optional extra (`store = ["zarr>=3.4.2", "healpy"]`), like `postproc`; zarr pinned to the
  version whose partial-shard and atomic-write behaviour was checked. rclone is an external
  binary (as for `PullCerraStage`).
- `PostProcStage.store: NetCDFStore | ZarrStore = NetCDFStore()`.
- Parallel postproc workers are started with `spawn`, not `fork`: zarr runs IO threads in the parent, and forking
  a multi-threaded process can deadlock. Workers import the stage's module afresh (a workspace's `pipeline.py` is
  on `sys.path` after activation). Blosc runs single-threaded (process parallelism already uses the CPUs).
- Scripts that run a stage with `run_parallel=True` need an `if __name__ == "__main__":` guard (spawned workers
  re-import the main module); the `wrf-massive` CLI and notebooks need nothing.
- `store pull` copies in two rclone passes: first the root and domain `zarr.json` and `sim_name` (to resolve
  `--sims` names to positions), then the rest (`- *.partial`, `+ <paths>`, `- **` filter file); afterwards it
  rebuilds the copy's consolidated metadata, so it lists only arrays that were copied, and merges the manifest
  (`.wrf-massive-pull.json`). `--dry-run` still copies the small first pass and reports `rclone size` of the rest.
- Readers open a domain group directly as a store (`xr.open_zarr("otprof.zarr/alps", consolidated=True)`):
  xarray's `group=` looks for consolidated metadata at the root, which the store deliberately does not have.
- The ML loader (dataset/sampler) lives outside wrf-massive and only depends on `read.py`.

## Other open questions

- Training sample pattern and inputs/targets (now only affects the tile size).
- Where training runs (GPU node reading GPFS, or local copy via `store pull`).
- Store location on Snellius (project storage) and its inode/space quota.
