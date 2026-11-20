# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interface between post-processing (reading, processing) and where its output goes (writing).

A store model is configuration only: it is validated with the pipeline, serialized to YAML and pickled to the
post-processing workers. Runtime state (open files, temp files, locks) lives in a `Writer`, created per input file
inside the worker by `PostProcStore.writer`; checks and locks for a whole simulation live in `PostProcStore.session`.
"""

from __future__ import annotations

import abc
import contextlib
import pathlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pydantic
import xarray as xr

if TYPE_CHECKING:
    from wrf_massive.base import Simulation, Stage
    from wrf_massive.store.frames import FrameTemplate


class StoreStage(Protocol):
    """What a store needs from a stage that writes into it (`PostProcStage`, `IngestStage`)."""

    name: str
    store: PostProcStore

    def get_work_dir(self, s: Simulation, create: bool = True) -> pathlib.Path: ...

    def stored_vars(self, s: Simulation) -> list[str]:
        """Names of the variables the stage writes for sim `s`."""

    def store_template(self, s: Simulation) -> FrameTemplate | None:
        """One output frame of `s` plus its file structure, to lay out a store (None: `s` has no input yet)."""

    def input_static(self, f: pathlib.Path, names: list[str]) -> dict[str, np.ndarray]:
        """Static (y, x) fields `names` of input file `f` as the stage outputs them (e.g. cropped); missing ones
        are skipped. Used to check that an input belongs to the store's grid."""


class Writer(abc.ABC):
    """Writes the processed frames of one input file. Created by `PostProcStore.writer`, used inside its context."""

    @abc.abstractmethod
    def pending_frames(self, n: int) -> list[int]:
        """Frame indices out of the input file's `n` frames that still have to be processed and written."""

    @abc.abstractmethod
    def write(self, ds: xr.Dataset, i0: int) -> None:
        """Write `ds`, holding one or more consecutive frames that start at frame `i0` of the input file."""


class PostProcStore(pydantic.BaseModel, abc.ABC):
    """Where post-processed frames go. Subclasses set a unique `kind` literal (the discriminator in configs)."""

    model_config = pydantic.ConfigDict(extra="forbid")  # a typo in a store option must not be silently ignored

    kind: str

    @contextlib.contextmanager
    def session(self, stage: Stage, s: Simulation, inputs: list[pathlib.Path]) -> Iterator[None]:
        """Context around processing all `inputs` of simulation `s`, entered once in the parent process before any
        input is processed: checks that everything can be written (fail fast) and holds locks for the whole run."""
        yield

    @abc.abstractmethod
    def is_done(self, stage: Stage, s: Simulation, inputs: list[pathlib.Path]) -> bool:
        """Whether the output of every input file of simulation `s` is complete."""

    @abc.abstractmethod
    @contextlib.contextmanager
    def writer(self, stage: Stage, s: Simulation, f: pathlib.Path) -> Iterator[Writer]:
        """Context with the `Writer` for input file `f`. Written frames are committed by `Writer.write` (zarr) or
        when the context is left without an exception (netCDF: the temp file is renamed)."""
