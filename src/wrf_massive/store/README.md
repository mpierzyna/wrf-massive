<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Stores: where post-processed output goes

A post-processing stage only reads and processes; its `store` decides where the result goes:

- `NetCDFStore` (default): one netCDF file per `wrfout` file in the stage's work dir.
- `ZarrStore`: one zarr store for many domains, laid out for machine learning. Each domain is a group with arrays
  `(sim, lead_time, bottom_top, y, x)`: native grid and model levels, no spin-up, one file (shard) per variable and
  input file, chunks of full columns of 64 × 64 grid points. Jobs write their frames straight into the store, in
  parallel and resuming after a crash; lossy encodings per variable save space with a known error bound. A HEALPix
  index of every grid point pairs the data with other datasets.

The zarr store needs the `store` extra (`wrf-massive[postproc,store]`). Design, layout and benchmarks:
[DESIGN.md](DESIGN.md).

## Writing into a zarr store

```python
# pipeline.py
from wrf_massive.stages.postproc.cn2 import Cn2PostProcStage
from wrf_massive.store.zarr_store import BitRound, FixedScaleOffset, ZarrStore

cn2 = Cn2PostProcStage(
    work_dir="4_postproc",
    wrfout_dir="3_wrf",
    domain=1,
    store=ZarrStore(
        path="/project/otprof.zarr",  # shared by all domains (relative paths: to the workspace root)
        domain="alps",  # group in the store
        static_vars=["HGT", "LANDMASK"],  # time-invariant: stored once
        encoding={"th": FixedScaleOffset(offset=300, step=1e-3), "cn2": BitRound(keepbits=10)},  # default: lossless
    ),
    run_parallel=True,
    resources=...,
)
```

```bash
wrf-massive store init-domain --stage cn2 sim_*   # once, before post-processing: lays out the domain and sims
wrf-massive submit --stages cn2 sim_*             # post-process into the store
```

`init-domain` writes metadata only and takes the layout from one processed frame of the first sim with output, so at
least one sim must have run WRF. Run it again to append new sims; from another workspace, it adds that domain to the
same store. Before writing, each job checks that its sim is initialized and its files fit the store.

## Ingesting existing netCDF output

Output already post-processed into netCDF is copied into the store with an `IngestStage`, which runs like any stage:

```python
from wrf_massive.store.ingest import IngestStage
from wrf_massive.store.netcdf import NetCDFStore

ingest = IngestStage(
    work_dir="5_ingest",  # holds only a lock
    src=NetCDFStore(file_suffix="cn2"),  # the netCDF files to read; root="/old/workspace" for another workspace's sims
    src_dir="4_postproc",  # work dir of the stage that wrote them
    store=ZarrStore(path="/project/otprof.zarr", domain="nl", static_vars=["HGT", "LANDMASK"]),
    run_parallel=True,
    resources=...,
)
```

```bash
wrf-massive store init-domain --stage ingest sim_*   # layout from the first netCDF file
wrf-massive submit --stages ingest sim_*             # or `run` locally
```

## CLI

```bash
wrf-massive store init-domain --stage cn2 sim_*         # create store/domain, append sims (metadata only)
wrf-massive --debug store init-domain --stage cn2 sim_* # dry run: only report what would be done
wrf-massive store verify --stage cn2                    # per domain: sims, frames written, size
wrf-massive store verify --path /project/otprof.zarr    # any store or copy, no workspace needed
wrf-massive store verify --path otprof.zarr --clean     # also remove leftovers of killed writes (nothing writing!)
wrf-massive store pull hpc:/project/otprof.zarr ~/data/otprof.zarr \
    --domain alps --vars cn2,z,HGT --sims 'sim_2017-*'   # partial copy with rclone (--rclone-config, --dry-run)
```

## Reading

```python
import numpy as np
from wrf_massive.store.zarr_store.read import hp_index, open_domain, select_cells

ds = open_domain("/project/otprof.zarr", "alps")  # lazy xarray dataset, sims sorted by begin
print(ds["sim_name"].values, ds["lead_time"].values)

frame = ds.isel(sim=0, lead_time=12)  # one sim, 12 h after the end of the spin-up
assert frame["written"]               # only written frames hold data (the others read as NaN)
log_cn2 = np.log10(frame["cn2"])      # (bottom_top, y, x); the store holds physical values
hagl = frame["z"] - ds["HGT"]         # height above ground of every model level

cells = hp_index(ds, 8)                                   # ERA5-sized HEALPix cell (order 8) of every grid point
in_cell = select_cells(ds, [cells.values[100, 200]], 8)  # (y, x) mask: the grid points inside that cell
columns = frame["cn2"].where(in_cell, drop=True)          # their Cn² columns
```

`open_domain(..., chunks=None)` reads without dask (each access goes straight to zarr). On a copy made by
`store pull`, `open_domain` exposes only the sims and variables the copy holds, so anything else raises instead of
reading as NaN.

Scripts that run a stage with `run_parallel=True` need an `if __name__ == "__main__":` guard: workers are started
with `spawn` and import the main module again (the `wrf-massive` CLI and notebooks need nothing).
