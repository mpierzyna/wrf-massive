# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`store pull`: copy part of a store (domains, variables, sims) with rclone, and record it in a manifest."""

from __future__ import annotations

import fnmatch
import json
import pathlib
import tempfile
import warnings

import zarr

from wrf_massive.log import get_logger
from wrf_massive.stages.utils import run_cmd_logged
from wrf_massive.store.zarr_store import layout as L
from wrf_massive.store.zarr_store.read import MANIFEST, read_manifest

logger = get_logger("store.pull")


def pull(
    src: str,
    dst: pathlib.Path,
    domains: list[str],
    variables: list[str] | None = None,
    sims: list[str] | None = None,
    rclone_config: pathlib.Path | None = None,
    transfers: int = 8,
    dry_run: bool = False,
) -> list[str]:
    """Copy `domains` of store `src` (an rclone path, `remote:path` or local) to the local dir `dst`.

    Metadata, coordinates, static fields and the `written` mask are always copied; of the data arrays only
    `variables` (default: all) and the shards of `sims` (sim names, globs allowed; default: all). The domain's
    consolidated metadata is rebuilt to list only what was copied, and a manifest in `dst` records it for the readers.
    With `dry_run`, only the domains' metadata is copied (needed to resolve sim names) and the size of the rest is
    reported. Returns a report.
    """
    dst = pathlib.Path(dst).absolute()
    dst.mkdir(parents=True, exist_ok=True)
    rclone = ["rclone", *(["--config", str(rclone_config)] if rclone_config else [])]
    report = []

    # 1. Root and domain metadata (consolidated, so it lists every array of the domain) and the sim names
    _copy(
        rclone,
        src,
        dst,
        ["/zarr.json", *(r for d in domains for r in (f"/{d}/zarr.json", f"/{d}/sim_name/**"))],
        transfers,
    )
    plan = {}
    for d in domains:
        if not (dst / d / "zarr.json").exists():
            raise KeyError(f"Domain '{d}' is not in {src}.")
        group = zarr.open_group(store=str(dst), path=d, mode="r", use_consolidated=True)
        arrays = sorted(group.array_keys())
        data = L.data_vars(group)
        if variables is not None and set(variables) - set(data):
            raise KeyError(f"'{d}' has no variable(s) {sorted(set(variables) - set(data))}; it has {data}.")
        names = list(group["sim_name"][:]) if "sim_name" in arrays else []
        plan[d] = dict(arrays=arrays, data=data, vars=variables, sim_names=names, sims=_select_sims(d, names, sims))

    # 2. Everything else of the selection
    rules = []
    for d, p in plan.items():
        selected = p["data"] if p["vars"] is None else p["vars"]
        for a in p["arrays"]:
            if a not in p["data"]:
                rules.append(f"/{d}/{a}/**")  # coordinates, static fields, mask: always complete
            elif a in selected:
                rules.append(f"/{d}/{a}/zarr.json")
                if p["sims"] is None:
                    rules.append(f"/{d}/{a}/c.*")
                else:
                    rules += [f"/{d}/{a}/c.{i}.*" for i in p["sims"]]
    size = json.loads(_rclone(rclone, ["size", src, "--json"], rules))
    report.append(f"{'Would copy' if dry_run else 'Copying'} {size['count']} file(s), {size['bytes'] / 1e9:.2f} GB")
    if dry_run:
        return report
    _copy(rclone, src, dst, rules, transfers)

    # 3. Local consolidated metadata lists only what is there; manifest for the readers
    manifest = read_manifest(dst) or {}
    for d, p in plan.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
            zarr.consolidate_metadata(store=str(dst), path=d)
        sims_held = None if p["sims"] is None else [p["sim_names"][i] for i in p["sims"]]
        manifest[d] = _merge(manifest.get(d), {"vars": p["vars"], "sims": sims_held})
        n_sims = len(p["sim_names"]) if sims_held is None else len(sims_held)
        n_vars = len(p["data"]) if p["vars"] is None else len(p["vars"])
        report.append(f"{d}: {n_vars} of {len(p['data'])} variable(s), {n_sims} of {len(p['sim_names'])} sim(s)")
    (dst / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return report


def _select_sims(domain: str, names: list[str], patterns: list[str] | None) -> list[int] | None:
    """Positions of the sims matching `patterns` (None: all sims)."""
    if patterns is None:
        return None
    selected = set()
    for pat in patterns:
        hits = [i for i, n in enumerate(names) if fnmatch.fnmatchcase(n, pat)]
        if not hits:
            raise KeyError(f"No sim of '{domain}' matches '{pat}'.")
        selected |= set(hits)
    return sorted(selected)


def _merge(old: dict | None, new: dict) -> dict:
    """What a copy holds after another pull (None means "all")."""
    if old is None:
        return new
    return {
        k: None if old.get(k) is None or new[k] is None else sorted(set(old[k]) | set(new[k])) for k in ("vars", "sims")
    }


def _filter_file(rules: list[str]) -> pathlib.Path:
    """rclone filter file: never temp files of interrupted writes, the given paths, nothing else."""
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="wrf_massive_pull_", delete=False)
    f.write("\n".join(["- *.partial", *(f"+ {r}" for r in rules), "- **"]) + "\n")
    f.close()
    return pathlib.Path(f.name)


def _rclone(rclone: list[str], args: list[str], rules: list[str]) -> str:
    f = _filter_file(rules)
    try:
        return run_cmd_logged(
            [*rclone, *args, "--filter-from", str(f)], logger, msg=f"running rclone {args[0]}", capture_stdout=True
        )
    finally:
        f.unlink()


def _copy(rclone: list[str], src: str, dst: pathlib.Path, rules: list[str], transfers: int) -> None:
    _rclone(rclone, ["copy", src, str(dst), "--transfers", str(transfers)], rules)
