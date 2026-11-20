# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frame times of input files (wrfout and post-processed netCDF), read from the file header only."""

from __future__ import annotations

import dataclasses
import datetime
import pathlib

import netCDF4
import numpy as np
import xarray as xr


def frame_times(f: pathlib.Path) -> np.ndarray:
    """Valid times (datetime64[s]) of the frames in `f`: WRF's `Times` or a CF `time` variable (also scalar)."""
    with netCDF4.Dataset(f) as nc:
        if "Times" in nc.variables:  # wrfout: char array (Time, DateStrLen)
            strings = netCDF4.chartostring(nc["Times"][:])
            return np.array(
                [np.datetime64(datetime.datetime.strptime(t, "%Y-%m-%d_%H:%M:%S")) for t in strings], "M8[s]"
            )
        if "time" in nc.variables:  # post-processed netCDF
            t = nc["time"]
            dates = netCDF4.num2date(
                np.atleast_1d(t[:]),
                t.units,
                getattr(t, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            return np.array([np.datetime64(d) for d in dates], "M8[s]")
    raise ValueError(f"{f}: neither a 'Times' nor a 'time' variable.")


def lead_indices(times: np.ndarray, begin: datetime.datetime, interval: np.timedelta64) -> np.ndarray:
    """Index of each frame time on the lead time axis that starts at `begin` with step `interval`."""
    offset = times.astype("M8[s]") - np.datetime64(begin, "s")
    idx, rest = np.divmod(offset, interval.astype("m8[s]"))
    if np.any(rest != np.timedelta64(0, "s")):
        raise ValueError(f"Frame times {times} are not on the {interval} grid starting at {begin}.")
    return idx.astype(int)


@dataclasses.dataclass
class FrameTemplate:
    """What a store needs to lay out a domain before any data is written, taken from one sim's output."""

    ds: xr.Dataset  # one processed frame, as `Writer.write` would receive it
    frames_per_file: int  # frames per input file: the shard length along lead time
    interval: np.timedelta64  # time between frames
    source: pathlib.Path  # file the template was taken from (for messages)
