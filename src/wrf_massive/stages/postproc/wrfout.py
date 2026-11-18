# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading variables from wrfout files with wrf-python."""

import pathlib

import netCDF4
import numpy as np
import xarray as xr

try:
    import wrf
except ImportError as e:
    raise ImportError(
        "The postprocessing stages need wrf-python, which is installed with the `postproc` extra of wrf-massive, "
        'e.g. `uv add "wrf-massive[postproc] @ git+https://github.com/mpierzyna/wrf-massive@<tag>"` '
        "(see the installation instructions in the README)."
    ) from e

from wrf_massive.log import get_logger
from wrf_massive.stages.postproc.variables import DERIVED_VARS, TVarList, is_wrf_field, var_name

logger = get_logger("stages.postproc")


def normalize_fill_values(v_data: xr.DataArray) -> xr.DataArray:
    """Replace declared fill/missing sentinels with NaN and drop the corresponding attributes.

    wrf-python stamps `_FillValue`/`missing_value` (float64 1e20) onto some float32 variables (e.g. `p`).
    xarray then warns about conflicting fill values on every read and silently decodes genuine 1e20 values
    to NaN. Normalising here means the returned dataset is plain NaN-based and safe to write and re-read.
    """
    fill_values = [v_data.attrs.pop(k, None) for k in ("_FillValue", "missing_value")]
    for k in ("_FillValue", "missing_value"):
        v_data.encoding.pop(k, None)

    if not np.issubdtype(v_data.dtype, np.floating):
        # Non-float data cannot hold NaN, so only the attributes are dropped.
        return v_data

    # Replace sentinels in-place (avoids copying these potentially huge arrays and preserves dtype)
    arr = v_data.values
    for fill in fill_values:
        if fill is None:
            continue
        try:
            if not np.isfinite(fill):
                continue  # NaN fill value: nothing to replace
        except (TypeError, ValueError):
            logger.warning(f"Ignoring non-numeric fill value '{fill}' of variable '{v_data.name}'.")
            continue
        arr[arr == fill] = np.nan

    return v_data


def read_var(wrfin: netCDF4.Dataset, name: str, destagger_dim: str | None = None) -> xr.DataArray:
    """Read variable `name` for all time steps from an open wrfout file and make it storable.

    A WRF field (ALL-CAPS) is looked up case-insensitively in the file. A derived variable is computed by wrf-python as
    defined in `DERIVED_VARS`. Either way, the result is returned under `name`.
    """
    if is_wrf_field(name):
        getvar_name = next((v for v in wrfin.variables if v.upper() == name), name)
        index = None
    else:
        getvar_name, _, index = DERIVED_VARS[name]

    # Get variable for all timesteps (ATTENTION! This is heavy on memory!)
    logger.info(f"Reading {name}... ")
    v_data = wrf.getvar(wrfin, getvar_name, timeidx=wrf.ALL_TIMES)
    if index is not None:  # e.g. u_met is uvmet[0]; drop=True leaves no scalar coordinate of the stacking dimension
        v_data = v_data.isel({v_data.dims[0]: index}, drop=True)

    if destagger_dim:
        destagger_dim_ind = v_data.dims.index(destagger_dim)
        logger.info(f"Destaggering along {destagger_dim} ({destagger_dim_ind})... ")
        v_data = wrf.destagger(v_data, destagger_dim_ind, meta=True)

    # Serialise object attributes for later netcdf storage
    v_data.attrs["projection"] = v_data.attrs["projection"].proj4()
    v_data = v_data.drop_vars("latlon_coord", errors="ignore")
    v_data = normalize_fill_values(v_data)
    return v_data.rename(name)


def load_wrfout(f: pathlib.Path | str, vars_to_extract: TVarList) -> xr.Dataset:
    """Load the given variables (WRF fields or wrf-python derived variables) from a wrfout file into a dataset."""
    # wrf-python reads eagerly, so all extracted data stays valid once the file is closed again.
    with netCDF4.Dataset(f) as wrf_file:
        res = {}
        for v in vars_to_extract:
            name = var_name(v)
            assert name not in res, f"Variable {name} already in results!"  # this shouldn't happen
            res[name] = read_var(wrf_file, name, destagger_dim=v[1] if isinstance(v, tuple) else None)
            logger.info("Done!")

        # Collect everything into a single dataset
        res_ds = xr.Dataset(res)
        res_ds = res_ds.drop_vars("latlon_coord", errors="ignore")

        # Copy attributes from original file
        for attr in wrf_file.ncattrs():
            res_ds.attrs[attr] = wrf_file.getncattr(attr)

    # Rename time
    res_ds = res_ds.rename({"Time": "time"})

    return res_ds
