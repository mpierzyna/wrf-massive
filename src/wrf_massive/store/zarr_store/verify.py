# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`store verify`: what a store (or a partial copy) holds, and cleanup of interrupted writes."""

from __future__ import annotations

import pathlib

import numpy as np

from wrf_massive.store.zarr_store import layout as L
from wrf_massive.store.zarr_store.read import read_manifest


def verify(
    path: pathlib.Path, domains: list[str] | None = None, sims: list[str] | None = None, clean: bool = False
) -> list[str]:
    """Report per domain: sims, frames written (and which sims are incomplete), variables and size on disk. Lists
    stray `*.partial` files of interrupted writes and, with `clean`, removes them (only while nothing writes)."""
    path = pathlib.Path(path).absolute()
    if not (path / "zarr.json").exists():
        raise FileNotFoundError(f"{path} is not a store.")
    root = L.open_root(path)
    report = [f"{path}  (schema {root.attrs.get('schema_version')})"]
    manifest = read_manifest(path)

    names = sorted(root.group_keys())
    for d in domains or names:
        if d not in names:
            raise KeyError(f"Domain '{d}' is not in {path}; it has {names}.")
        group = L.open_domain(path, d)
        sim_names = np.asarray(group["sim_name"][:])
        written = group[L.WRITTEN][:]
        rows = np.arange(len(sim_names)) if sims is None else np.flatnonzero(np.isin(sim_names, sims))
        if sims is not None and len(rows) < len(set(sims)):
            missing = sorted(set(sims) - set(sim_names))
            report.append(f"  {d}: not initialized: {', '.join(missing)}")
        w = written[rows]
        incomplete = [str(sim_names[i]) for i, row in zip(rows, w) if not row.all()]
        size = sum(f.stat().st_size for f in (path / d).rglob("*") if f.is_file())
        line = (
            f"  {d:10s} {len(rows):4d} sims  {int(w.sum())}/{w.size} frames written  "
            f"{len(L.data_vars(group))} vars  {size / 1e9:.2f} GB on disk"
        )
        if incomplete:
            shown = ", ".join(incomplete[:5]) + (", ..." if len(incomplete) > 5 else "")
            line += f"  incomplete: {len(incomplete)} sims ({shown})"
        report.append(line)

    if manifest is not None:
        report.append("  local copy (manifest; `written` describes the source store):")
        for d, held in sorted(manifest.items()):
            v = "all vars" if held.get("vars") is None else "vars " + " ".join(held["vars"])
            s = "all sims" if held.get("sims") is None else f"{len(held['sims'])} sims"
            report.append(f"    {d}: {v}, {s}")

    partial = sorted(path.rglob("*.partial"))
    report.append(f"  stray .partial files: {len(partial)}")
    if clean:
        for f in partial:
            f.unlink()
        if partial:
            report.append(f"  removed {len(partial)} .partial file(s)")
    else:
        report += [f"    {f.relative_to(path)}" for f in partial[:10]]
    return report
