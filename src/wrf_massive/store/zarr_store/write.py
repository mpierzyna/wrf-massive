# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Writing into a domain group: preflight checks + sim lock (`session`), the frame writer and `is_done`.

Workers only write chunk data, never metadata. Parallel safety rests on one writer per shard: one worker per input
file, input files aligned with shards (checked in `session`) and one job per sim (lock in `session`).
"""

from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import netCDF4
import numcodecs.blosc
import numpy as np
import xarray as xr

from wrf_massive.log import get_logger
from wrf_massive.store.base import Writer
from wrf_massive.store.frames import frame_times, lead_indices
from wrf_massive.store.zarr_store import layout as L
from wrf_massive.store.zarr_store.store import ZarrStore

if TYPE_CHECKING:
    import zarr

    from wrf_massive.base import Simulation
    from wrf_massive.store.base import StoreStage

logger = get_logger("store.zarr")

SIM_LOCK = ".zarr_store.lock"  # in the stage's work dir: outside the store, so copies of the store stay clean


def _open(cfg: ZarrStore, stage_name: str, mode: str = "r") -> zarr.Group:
    path = cfg.abs_path
    if not (path / cfg.domain / "zarr.json").exists():
        raise RuntimeError(
            f"Domain '{cfg.domain}' is not in the store {path}. Run `wrf-massive store init-domain --stage "
            f"{stage_name} <sim dirs>` first."
        )
    group = L.open_domain(path, cfg.domain, mode=mode)
    if group.attrs.get("schema_version") != L.SCHEMA_VERSION:
        raise RuntimeError(
            f"Domain '{cfg.domain}' has schema version {group.attrs.get('schema_version')}, this wrf-massive writes "
            f"version {L.SCHEMA_VERSION}."
        )
    return group


def _sim_index(group: zarr.Group, s: Simulation, cfg: ZarrStore, stage_name: str) -> int:
    idx = L.sim_index(group, s.name)
    if idx is None:
        raise RuntimeError(
            f"sim '{s.name}' is not initialized in domain '{cfg.domain}' of {cfg.abs_path}. Run `wrf-massive store "
            f"init-domain --stage {stage_name} {s.sim_dir}` first."
        )
    return idx


def _leads(group: zarr.Group, s: Simulation, f: pathlib.Path) -> np.ndarray:
    return lead_indices(frame_times(f), s.begin, np.timedelta64(group.attrs["interval_s"], "s"))


@contextlib.contextmanager
def session(cfg: ZarrStore, stage: StoreStage, s: Simulation, inputs: list[pathlib.Path]) -> Iterator[None]:
    """Preflight checks (nothing is written if one fails), then hold the sim lock while the inputs are processed."""
    group = _open(cfg, stage.name)

    # 1. The domain matches the stage: same variables, static vars, tile and encodings as at init-domain
    expected = set(stage.stored_vars(s)) - set(cfg.static_vars) - set(L.COORDS_2D)
    have = set(L.data_vars(group))
    if expected != have:
        raise RuntimeError(
            f"Domain '{cfg.domain}' does not match stage '{stage.name}': variables only in the stage "
            f"{sorted(expected - have)}, only in the store {sorted(have - expected)}."
        )
    if set(group.attrs["static_vars"]) != set(cfg.static_vars):
        raise RuntimeError(f"static_vars {cfg.static_vars} differ from the store's {group.attrs['static_vars']}.")
    if list(group.attrs["tile"]) != list(cfg.tile):
        raise RuntimeError(f"tile {cfg.tile} differs from the store's {group.attrs['tile']}.")
    differs = {
        v: (group[v].attrs.get("encoding"), cfg.encoding_of(v).model_dump())
        for v in sorted(have)
        if group[v].attrs.get("encoding") != cfg.encoding_of(v).model_dump()
    }
    if differs:
        raise RuntimeError(
            "Encodings differ from the store (changing an encoding means rewriting the array): "
            + "; ".join(f"{v}: store {a}, stage {b}" for v, (a, b) in differs.items())
        )

    # 2. The sim is initialized
    _sim_index(group, s, cfg, stage.name)

    # 3. The inputs fit: on the lead time grid, after the spin-up, each file within one shard and no shard twice
    n_lead, shard_len = group.attrs["n_lead"], group.attrs["shard_len"]
    shards: dict[int, pathlib.Path] = {}
    for f in inputs:
        leads = _leads(group, s, f)
        if leads[0] < 0:
            raise RuntimeError(
                f"{f.name} has frames before Simulation.begin ({s.begin}): the store holds no spin-up. Is the "
                f"warmup a multiple of the output interval and `discard_warmup` set?"
            )
        if leads[-1] >= n_lead:
            raise RuntimeError(f"{f.name} has frames after Simulation.end ({s.end}).")
        shard = int(leads[0]) // shard_len
        if not (leads[0] % shard_len == 0 and np.all(np.diff(leads) == 1) and int(leads[-1]) // shard_len == shard):
            raise RuntimeError(
                f"{f.name} (lead times {leads.tolist()}) is not aligned with the {shard_len}-frame shards."
            )
        if shard in shards:
            raise RuntimeError(f"{f.name} and {shards[shard].name} would both write shard {shard}.")
        shards[shard] = f
    if inputs:
        _check_static(group, cfg, inputs[0])

    # 4. No other job writes this sim (one writer per shard)
    with L.exclusive_lock(stage.get_work_dir(s) / SIM_LOCK, f"Writing sim '{s.name}' to the store"):
        yield


def _check_static(group: zarr.Group, cfg: ZarrStore, f: pathlib.Path) -> None:
    """Grid and static fields of input `f` equal the stored ones (fields not in the file itself are skipped)."""
    with netCDF4.Dataset(f) as nc:
        nc.set_auto_mask(False)
        for v in [*L.COORDS_2D, *cfg.static_vars]:
            if v not in nc.variables:
                continue
            values = np.asarray(nc[v][:], dtype="float32")
            values = values.reshape(-1, *values.shape[-2:])[0]  # first frame
            stored = group[v][:]
            if values.shape != stored.shape or not np.array_equal(values, stored, equal_nan=True):
                raise RuntimeError(f"'{v}' of {f.name} differs from the store's (another grid or domain?).")


def is_done(cfg: ZarrStore, s: Simulation, inputs: list[pathlib.Path]) -> bool:
    """Every frame of every input file is written."""
    path = cfg.abs_path
    if not (path / cfg.domain / "zarr.json").exists():
        return False
    group = L.open_domain(path, cfg.domain)
    idx = L.sim_index(group, s.name)
    if idx is None:
        return False
    row = group[L.WRITTEN][idx]
    try:
        leads = np.concatenate([_leads(group, s, f) for f in inputs]) if inputs else np.array([], int)
    except ValueError:
        return False  # frames off the lead time grid: never written
    return bool(np.all((leads >= 0) & (leads < len(row)))) and bool(np.all(row[leads]))


class ZarrWriter(Writer):
    """Writes the frames of one input file into region `[sim, lead_time]` of each data array, then the mask."""

    def __init__(self, cfg: ZarrStore, stage_name: str, s: Simulation, f: pathlib.Path):
        # Each worker process writes one file: Blosc threads on top would oversubscribe the CPUs. Set here, not at
        # import, so readers (e.g. an ML data loader) keep Blosc's threads.
        numcodecs.blosc.use_threads = False
        self.cfg, self.f = cfg, f
        self.group = _open(cfg, stage_name, mode="r+")
        self.idx = _sim_index(self.group, s, cfg, stage_name)
        self.times = frame_times(f)
        self.leads = _leads(self.group, s, f)
        self.static = set(self.group.attrs["static_vars"])
        self.arrays = {v: self.group[v] for v in L.data_vars(self.group)}

    def pending_frames(self, n: int) -> list[int]:
        if n != len(self.leads):
            raise RuntimeError(f"{self.f.name}: {n} frames requested, the file has {len(self.leads)}.")
        written = self.group[L.WRITTEN][self.idx, int(self.leads[0]) : int(self.leads[-1]) + 1]
        return [i for i in range(n) if not written[i]]

    def write(self, ds: xr.Dataset, i0: int) -> None:
        n = ds.sizes["time"]
        times = ds["time"].values.astype("M8[s]")
        if not np.array_equal(times, self.times[i0 : i0 + n]):
            raise RuntimeError(
                f"{self.f.name}: frames {i0}..{i0 + n - 1} have times {times}, the file {self.times[i0 : i0 + n]}."
            )
        names = set(ds.data_vars) - self.static - set(L.COORDS_2D)
        if names != set(self.arrays):
            raise RuntimeError(
                f"{self.f.name}: variables not in the store {sorted(names - set(self.arrays))}, store variables "
                f"missing {sorted(set(self.arrays) - names)}."
            )

        l0 = int(self.leads[i0])
        for v, arr in self.arrays.items():
            da = ds[v]
            dims = L.FRAME_DIMS.get(da.dims)
            if dims != tuple(arr.metadata.dimension_names):
                raise RuntimeError(
                    f"'{v}' has dims {da.dims}, the store expects {arr.metadata.dimension_names} "
                    f"(staggered dims must be destaggered in postproc)."
                )
            values = np.asarray(da.values, dtype=np.float32)
            self.cfg.encoding_of(v).check(v, values)
            # Edge tiles are padded with the NaN fill value, which FixedScaleOffset casts to int (cells outside the
            # grid, never read): silence that warning; real values were checked above
            with np.errstate(invalid="ignore"):
                arr[self.idx, l0 : l0 + n] = values
        # Commit: only now the frames count as written (a crash before leaves them False and they are redone)
        self.group[L.WRITTEN][self.idx, l0 : l0 + n] = True
