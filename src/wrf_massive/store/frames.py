# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frame times of input files (wrfout and post-processed netCDF), read from the file header only."""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
from collections.abc import Callable

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


HORIZONTAL_DIMS = ("south_north", "west_east", "south_north_stag", "west_east_stag")


def crop_xy(ds: xr.Dataset, n: int) -> xr.Dataset:
    """`ds` without the outer `n` cells on each side of every horizontal dim (`[n:-n]`), e.g. WRF's lateral boundary
    zone (`spec_bdy_width`), where the forcing is applied. Coordinates (XLAT/XLONG) are cropped alike."""
    if n == 0:
        return ds
    dims = [d for d in HORIZONTAL_DIMS if d in ds.dims]
    small = {d: ds.sizes[d] for d in dims if ds.sizes[d] <= 2 * n}
    if small:
        raise ValueError(f"crop={n} leaves no grid points along {small}.")
    out = ds.isel({d: slice(n, -n) for d in dims})
    out.attrs["crop"] = n
    return out


def read_static(f: pathlib.Path, names: list[str], crop: int = 0) -> dict[str, np.ndarray]:
    """First frame of the (y, x) fields `names` of input file `f` (wrfout or netCDF output), cropped like the
    stage's output; names not in the file are skipped."""
    out = {}
    with netCDF4.Dataset(f) as nc:
        nc.set_auto_mask(False)
        for v in names:
            if v in nc.variables:
                values = np.asarray(nc[v][:], dtype="float32")
                values = values.reshape(-1, *values.shape[-2:])[0]
                out[v] = values[crop:-crop, crop:-crop] if crop else values
    return out


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

    @classmethod
    def from_inputs(
        cls, inputs: list[pathlib.Path], first_frame: Callable[[pathlib.Path], xr.Dataset]
    ) -> FrameTemplate:
        """Template of a sim's input files: `first_frame(inputs[0])`, the largest number of frames per file and the
        interval between frames (from the first file with two of them)."""
        times = [frame_times(f) for f in inputs]
        steps = [t[1] - t[0] for t in times if len(t) > 1]
        if not steps:
            raise ValueError(f"None of the {len(inputs)} input file(s) has two frames to take the interval from.")
        return cls(
            ds=first_frame(inputs[0]),
            frames_per_file=max(len(t) for t in times),
            interval=steps[0],
            source=inputs[0],
        )
