from __future__ import annotations

import pathlib
from typing import List, Tuple

import netCDF4
import pandas as pd
import wrf
import xarray as xr

# Add string of variable names to be extracted.
# If destaggering needs to happen, add tuple with varname and destagger_dim.
WRF_DEFAULT_VARS: List[str | Tuple[str, str]] = [
    "z",
    "HGT",
    "uvmet",
    "wa",
    "th",
    "tk",
    "rh",
    "p",
    "PBLH",
    "LANDMASK",
    # "QRAIN",
    # "dbz",
    "slp",
    "T2",
    "TH2",
    "LH",
    "HFX",
    "ZNT",
    "Z0",
    "UST",
    "QKE",
    ("EL_PBL", "bottom_top_stag"),
    "TSQ",
]


def load_wrfout(
    f: pathlib.Path | str,
    vars_to_extract: List[str | Tuple[str, str]] | None,
    dt_warmup: str | None,
) -> xr.Dataset:
    wrf_file = netCDF4.Dataset(f)

    if vars_to_extract is None:
        vars_to_extract = WRF_DEFAULT_VARS

    # Start actual extraction
    res = {}
    for v in vars_to_extract:
        # Optionally: Extract dimension along which data will be destaggered
        if isinstance(v, tuple):
            v, destagger_dim = v
        else:
            destagger_dim = None

        # Get variable for all timesteps (ATTENTION! This is heavy on memory!)
        print(f"Reading {v}... ", end="")
        v_data = wrf.getvar(wrf_file, v, timeidx=wrf.ALL_TIMES)

        # Now destagger
        if destagger_dim:
            destagger_dim_ind = v_data.dims.index(destagger_dim)
            print(f"Destaggering along {destagger_dim} ({destagger_dim_ind})... ", end="")
            v_data = wrf.destagger(v_data, destagger_dim_ind, meta=True)

        # Serialise object attributes for later netcdf storage
        v_data.attrs["projection"] = v_data.attrs["projection"].proj4()
        v_data = v_data.drop_vars("latlon_coord", errors="ignore")

        # Store it. Maybe write to output file directly to release RAM
        assert v not in res, f"Variable {v} already in results!"  # this shouldn't happen
        res[v] = v_data
        print(f"Done!")

    # Collect everything into a single dataset
    res_ds = xr.Dataset(res)
    res_ds = res_ds.drop_vars("latlon_coord", errors="ignore")

    # Copy attributes from original file
    for attr in wrf_file.ncattrs():
        res_ds.attrs[attr] = wrf_file.getncattr(attr)

    ## Manually post-process some data
    # Split wind velocities into individual variables
    if "uvmet" in res_ds:
        res_ds["u_met"] = res_ds["uvmet"][0]
        res_ds["v_met"] = res_ds["uvmet"][1]
        res_ds = res_ds.drop_vars(["uvmet", "u_v"])

    # Rename vertical wind velocity
    if "wa" in res_ds:
        res_ds = res_ds.rename({"wa": "w"})

    # Rename time
    res_ds = res_ds.rename({"Time": "time"})

    # Remove warmup period
    if dt_warmup:
        dt_warmup = pd.to_timedelta(dt_warmup)
        res_ds = res_ds.sel(time=slice(res_ds.time[0] + dt_warmup, None))

    return res_ds
