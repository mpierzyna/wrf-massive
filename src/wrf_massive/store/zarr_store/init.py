# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`init-domain`: create the store and a domain group, and append sims (preallocation, metadata only)."""

from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import json
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import zarr

from wrf_massive.store.frames import FrameTemplate
from wrf_massive.store.zarr_store import layout as L
from wrf_massive.store.zarr_store.encoding import Lossless
from wrf_massive.store.zarr_store.store import ZarrStore

if TYPE_CHECKING:
    from wrf_massive.base import Simulation, Stage


_DROP_ATTRS = {"stagger", "coordinates", "projection", "MemoryOrder", "FieldType", "_FillValue", "missing_value"}


def zarr_store_of(stage: Stage) -> ZarrStore:
    """The `ZarrStore` a stage writes to, or a ValueError naming the stage."""
    store = getattr(stage, "store", None)
    if not isinstance(store, ZarrStore):
        raise ValueError(f"Stage '{stage.name}' does not write to a ZarrStore (store: {type(store).__name__}).")
    return store


def init_domain(stage: Stage, sims: list[Simulation], dry_run: bool = False) -> list[str]:
    """Add the domain of `stage`'s ZarrStore and the given sims to the store (see `wrf-massive store init-domain`).

    Creates the store if needed and the domain group from a template (one processed frame of the first given sim
    with output), then appends the sims not yet in the domain. Returns what was (or, with `dry_run`, would be) done.
    """
    cfg = zarr_store_of(stage)
    path = cfg.abs_path
    report: list[str] = []
    lock = L.exclusive_lock(L.domain_lock_path(path, cfg.domain), f"init-domain of '{cfg.domain}'")
    with lock if not dry_run else contextlib.nullcontext():
        # Store (root group)
        if not (path / "zarr.json").exists():
            report.append(f"Store {path} does not exist -> creating (schema {L.SCHEMA_VERSION})")
            if not dry_run:
                L.open_root(path, mode="w-").attrs.update(
                    schema_version=L.SCHEMA_VERSION, created_by=f"wrf-massive {_version()}"
                )
        else:
            _check_schema(L.open_root(path).attrs.asdict(), path)

        # Domain group
        sims = list({s.name: s for s in sims}.values())  # unique, in order
        if not (path / cfg.domain / "zarr.json").exists():
            template, s_t = _find_template(stage, sims)
            spec = _DomainSpec(cfg, template, s_t)
            report.append(f"Domain '{cfg.domain}' does not exist -> creating from template {template.source}")
            report.append(f"-> {spec.summary()}")
            if not dry_run:
                _create_domain(cfg, spec, template, s_t)
            existing, n_lead, interval = [], spec.n_lead, spec.interval_s
        else:
            group = L.open_domain(path, cfg.domain)
            existing = list(group["sim_name"][:])
            n_lead, interval = group.attrs["n_lead"], group.attrs["interval_s"]

        # Sims
        new = [s for s in sims if s.name not in existing]
        for s in new:
            n = (s.end - s.begin).total_seconds() / interval + 1
            if n != n_lead:
                raise ValueError(
                    f"{s.name}: {s.begin} .. {s.end} gives {n:g} lead times at {interval} s, the domain has {n_lead}."
                )
        skipped = [s.name for s in sims if s.name in existing]
        report.append(
            f"Appending {len(new)} sim(s)"
            + (f" ({len(skipped)} already present: {', '.join(skipped)})" if skipped else "")
        )
        if not dry_run:
            if new:
                _append_sims(L.open_domain(path, cfg.domain, mode="r+"), new)
            with warnings.catch_warnings():  # "not part of the Zarr format 3 specification" (yet): known
                warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
                zarr.consolidate_metadata(store=str(path), path=cfg.domain)
            report += _encoding_differences(cfg)
    return report


def _version() -> str:
    try:
        return importlib.metadata.version("wrf_massive")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _check_schema(attrs: dict, path) -> None:
    if attrs.get("schema_version") != L.SCHEMA_VERSION:
        raise RuntimeError(
            f"Store {path} has schema version {attrs.get('schema_version')}, this wrf-massive writes version "
            f"{L.SCHEMA_VERSION}."
        )


def _find_template(stage: Stage, sims: list[Simulation]) -> tuple[FrameTemplate, Simulation]:
    for s in sims:
        template = stage.store_template(s)
        if template is not None:
            return template, s
    raise ValueError(
        f"The domain does not exist yet and none of the {len(sims)} given sim(s) has output to take its layout from "
        f"(postproc: wrfout aux files; ingest: netCDF output). Include a sim that has."
    )


class _DomainSpec:
    """Shapes, dims and attrs of all arrays of a new domain group, derived from a template frame."""

    def __init__(self, cfg: ZarrStore, template: FrameTemplate, s: Simulation):
        ds = template.ds
        self.interval_s = int(template.interval / np.timedelta64(1, "s"))
        self.n_lead = int((s.end - s.begin).total_seconds() // self.interval_s + 1)
        self.shard_len = template.frames_per_file
        self.ny, self.nx = ds.sizes["south_north"], ds.sizes["west_east"]
        self.nz = ds.sizes.get("bottom_top")

        for c in L.COORDS_2D:
            if c not in ds.variables:
                raise ValueError(f"The template has no '{c}' (needed as static coordinate).")
        missing_static = [v for v in cfg.static_vars if v not in ds.data_vars]
        if missing_static:
            raise ValueError(f"static_vars {missing_static} are not in the stage's output.")
        unknown_enc = [v for v in cfg.encoding if v not in ds.data_vars or v in cfg.static_vars]
        if unknown_enc:
            raise ValueError(f"encoding given for {unknown_enc}, which are not time-dependent output variables.")

        self.data: dict[str, tuple[str, ...]] = {}  # name -> store dims
        self.static: list[str] = []
        for v in ds.data_vars:
            dims = ds[v].dims
            if v in cfg.static_vars:
                if dims[-2:] != L.FRAME_STATIC_DIMS or len(dims) > 3:
                    raise ValueError(f"static var '{v}' has dims {dims}, expected (time,) south_north, west_east.")
                self.static.append(v)
            elif dims in L.FRAME_DIMS:
                self.data[v] = L.FRAME_DIMS[dims]
            else:
                raise ValueError(
                    f"'{v}' has dims {dims}; the store holds (time, bottom_top, south_north, west_east) or "
                    f"(time, south_north, west_east). Destagger staggered dims in postproc, e.g. "
                    f"('{v}', '{next((d for d in dims if d.endswith('_stag')), 'bottom_top_stag')}'), or list a "
                    f"time-invariant variable in static_vars."
                )

    def summary(self) -> str:
        n3 = sum(len(d) == 5 for d in self.data.values())
        return (
            f"{len(self.data)} variables ({n3} 3D, {len(self.data) - n3} 2D), {len(self.static)} static, "
            f"grid {self.ny} x {self.nx}, {self.nz} levels, {self.n_lead} lead times every {self.interval_s} s, "
            f"{self.shard_len} per shard"
        )


def _create_domain(cfg: ZarrStore, spec: _DomainSpec, template: FrameTemplate, s: Simulation) -> None:
    ds = template.ds
    group = L.open_root(cfg.abs_path, mode="r+").create_group(cfg.domain)
    group.attrs.update(
        schema_version=L.SCHEMA_VERSION,
        domain=cfg.domain,
        interval_s=spec.interval_s,
        n_lead=spec.n_lead,
        shard_len=spec.shard_len,
        tile=list(cfg.tile),
        warmup_h=s.warmup_h,
        static_vars=list(cfg.static_vars),
        created_by=f"wrf-massive {_version()}",
        template=str(template.source),
        wrf_attrs=L.jsonable(ds.attrs),
    )
    ty, tx = cfg.tile
    sy, sx = math.ceil(spec.ny / ty) * ty, math.ceil(spec.nx / tx) * tx  # shards must be a multiple of chunks
    coords = "time " + " ".join(L.COORDS_2D)
    for v, dims in spec.data.items():
        enc = cfg.encoding_of(v)
        three_d = len(dims) == 5
        chunks = (1, 1, spec.nz, ty, tx) if three_d else (1, 1, ty, tx)
        shards = (1, spec.shard_len, spec.nz, sy, sx) if three_d else (1, spec.shard_len, sy, sx)
        shape = (0, spec.n_lead, *chunks[2:-2], spec.ny, spec.nx)
        attrs = {k: a for k, a in L.jsonable(ds[v].attrs).items() if k not in _DROP_ATTRS}
        group.create_array(
            v,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype="float32",
            fill_value=np.nan,
            dimension_names=dims,
            chunk_key_encoding=L.CHUNK_KEYS,
            attributes={**attrs, "coordinates": coords, "encoding": enc.model_dump()},
            **enc.codecs(),
        )

    # Static fields (incl. lat/lon), stored once
    lossless = Lossless()
    for v in [*L.COORDS_2D, *spec.static]:
        values = ds[v].values
        values = values.reshape(values.shape[-2:]) if values.ndim > 2 else values  # first (only) frame
        attrs = {k: a for k, a in L.jsonable(ds[v].attrs).items() if k not in _DROP_ATTRS}
        group.create_array(
            v, data=values.astype("float32"), dimension_names=L.STATIC_DIMS, attributes=attrs, **lossless.codecs()
        )
    group.create_array(
        "hp_ipix",
        data=_hp_ipix(ds["XLAT"].values, ds["XLONG"].values),
        dimension_names=L.STATIC_DIMS,
        attributes={
            "order": L.HP_ORDER,
            "indexing_scheme": "nested",
            "source": "XLAT/XLONG (float32, cast to float64)",
            "description": "HEALPix cell of each grid point; coarser order o: hp_ipix >> 2 * (order - o)",
        },
        **lossless.codecs(),
    )

    # Coordinates and per-sim metadata
    lead = np.arange(spec.n_lead, dtype="int64") * spec.interval_s
    group.create_array(
        "lead_time",
        data=lead,
        dimension_names=("lead_time",),
        attributes={"units": "seconds", "dtype": "timedelta64[s]"},
    )
    per_sim = dict(dimension_names=("sim",), chunks=(L.SIM_CHUNK,))
    time_attrs = {"units": L.TIME_UNITS, "calendar": "proleptic_gregorian"}
    group.create_array("sim_name", shape=(0,), dtype=str, fill_value="", **per_sim)
    group.create_array("sim_begin", shape=(0,), dtype="int64", fill_value=0, attributes=time_attrs, **per_sim)
    group.create_array("sim_settings", shape=(0,), dtype=str, fill_value="", **per_sim)
    group.create_array(
        "time",
        shape=(0, spec.n_lead),
        chunks=(L.SIM_CHUNK, spec.n_lead),
        dtype="int64",
        fill_value=0,
        dimension_names=("sim", "lead_time"),
        attributes={**time_attrs, "description": "valid time"},
    )
    # One chunk per (sim, shard): owned by the worker that writes the shard, like the data
    group.create_array(
        L.WRITTEN,
        shape=(0, spec.n_lead),
        chunks=(1, spec.shard_len),
        dtype="bool",
        fill_value=False,
        dimension_names=("sim", "lead_time"),
        chunk_key_encoding=L.CHUNK_KEYS,
        attributes={"description": "frame [sim, lead_time] completely written (commit mask)"},
    )


def _hp_ipix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    import healpy

    lat, lon = (a.reshape(a.shape[-2:]).astype("float64") for a in (lat, lon))
    return healpy.ang2pix(2**L.HP_ORDER, lon, lat, nest=True, lonlat=True).astype("uint64")


def _append_sims(group: zarr.Group, sims: list[Simulation]) -> None:
    n0 = group["sim_name"].shape[0]
    n1 = n0 + len(sims)
    for _, arr in group.arrays():
        dims = arr.metadata.dimension_names
        if dims and dims[0] == "sim":
            arr.resize((n1, *arr.shape[1:]))  # metadata only, existing chunks are untouched
    epoch = datetime.datetime(1970, 1, 1)
    begins = np.array([int((s.begin - epoch).total_seconds()) for s in sims], dtype="int64")
    group["sim_name"][n0:n1] = np.array([s.name for s in sims], dtype=object)
    group["sim_begin"][n0:n1] = begins
    group["sim_settings"][n0:n1] = np.array([json.dumps(s.settings, sort_keys=True) for s in sims], dtype=object)
    group["time"][n0:n1] = begins[:, None] + group["lead_time"][:][None, :]


def _encoding_differences(cfg: ZarrStore) -> list[str]:
    """Warnings for variables another domain of the store encodes differently."""
    root = L.open_root(cfg.abs_path)
    mine = L.open_domain(cfg.abs_path, cfg.domain)
    warnings = []
    for other in sorted(set(root.group_keys()) - {cfg.domain}):
        group = root[other]
        for v in set(L.data_vars(mine)) & set(L.data_vars(group)):
            a, b = mine[v].attrs.get("encoding"), group[v].attrs.get("encoding")
            if a != b:
                warnings.append(f"WARNING: '{v}' is encoded as {a} here but as {b} in domain '{other}'.")
    return warnings
