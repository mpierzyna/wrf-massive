# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`ZarrStore`: configuration of one domain group in a zarr store. zarr is imported only when the store is used."""

from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import pydantic

from wrf_massive.base import TPath
from wrf_massive.store.base import PostProcStore, Writer
from wrf_massive.store.zarr_store.encoding import Encoding, Lossless

if TYPE_CHECKING:
    from wrf_massive.base import Simulation, Stage


class ZarrStore(PostProcStore):
    """One domain group of a zarr store that holds the post-processed output of many domains (see DESIGN.md).

    The domain must be laid out with `wrf-massive store init-domain` before any sim is written. Frames are written
    straight into their region `[sim, lead_time]`; the `written` mask records which frames are complete.
    """

    kind: Literal["zarr"] = "zarr"
    path: TPath  # the store (shared by all domains), e.g. on project storage
    domain: str = pydantic.Field(pattern=r"^[A-Za-z0-9_\-]+$")  # name of the domain group
    tile: tuple[pydantic.PositiveInt, pydantic.PositiveInt] = (64, 64)  # horizontal (y, x) size of a chunk
    static_vars: list[str] = []  # time-invariant variables (e.g. HGT, LANDMASK): stored once, not per frame
    encoding: dict[str, Encoding] = {}  # per variable; variables not listed are stored lossless

    def encoding_of(self, name: str) -> Encoding:
        return self.encoding.get(name, Lossless())

    @property
    def abs_path(self) -> pathlib.Path:
        """The store path made absolute (relative paths are relative to the workspace root, the cwd of commands)."""
        return pathlib.Path(self.path).absolute()

    @contextlib.contextmanager
    def session(self, stage: Stage, s: Simulation, inputs: list[pathlib.Path]) -> Iterator[None]:
        from wrf_massive.store.zarr_store.write import session

        with session(self, stage, s, inputs):
            yield

    def is_done(self, stage: Stage, s: Simulation, inputs: list[pathlib.Path]) -> bool:
        from wrf_massive.store.zarr_store.write import is_done

        return is_done(self, s, inputs)

    @contextlib.contextmanager
    def writer(self, stage: Stage, s: Simulation, f: pathlib.Path) -> Iterator[Writer]:
        from wrf_massive.store.zarr_store.write import ZarrWriter

        yield ZarrWriter(self, stage.name, s, f)
