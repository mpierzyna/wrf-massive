# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interface between post-processing (reading, processing) and where its output goes (writing).

A store model is configuration only: it is validated with the pipeline, serialized to YAML and pickled to the
post-processing workers. Runtime state (open files, temp files, locks) lives in a `Writer`, created per input file
inside the worker by `PostProcStore.writer`.
"""

from __future__ import annotations

import abc
import contextlib
import pathlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pydantic
import xarray as xr

if TYPE_CHECKING:
    from wrf_massive.base import Simulation, Stage


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

    def preflight(self, stage: Stage, s: Simulation, inputs: list[pathlib.Path]) -> None:
        """Check before any input is processed that everything can be written. Runs once, in the parent process."""

    @abc.abstractmethod
    def is_done(self, stage: Stage, s: Simulation, inputs: list[pathlib.Path]) -> bool:
        """Whether the output of every input file of simulation `s` is complete."""

    @abc.abstractmethod
    @contextlib.contextmanager
    def writer(self, stage: Stage, s: Simulation, f: pathlib.Path) -> Iterator[Writer]:
        """Context with the `Writer` for input file `f`. Leaving it without an exception commits the output."""
