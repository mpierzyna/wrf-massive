# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Workspaces: a directory holding an env file, `pipeline.py`, `simulations.py` and namelist templates.

The workspace root is the directory containing the env file (default `env.yaml`). It is found by searching the
current directory and its parents, so the `wrf-massive` CLI works from anywhere inside a workspace.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import logging
import os
import pathlib
import sys
from types import ModuleType
from typing import Any, Dict, List

from wrf_massive.base import Pipeline
from wrf_massive.config import yaml_to_dict
from wrf_massive.log import get_logger

logger = get_logger()

DEFAULT_ENV_FILE = "env.yaml"
ENV_FILE_ENVVAR = "WRF_MASSIVE_ENV_FILE"  # read by the CLI's `--env-file` and by `load_env`
PIPELINE_MODULE = "pipeline"  # `pipeline.py` in the workspace root
DEFAULT_LOG_LEVEL = "info"
LOG_FORMAT = "%(asctime)s %(levelname)s:%(name)s:%(message)s"

# Packaged files copied by `init_workspace`, mapped to their name in the new workspace. Files are stored under a
# different name in the package where the real name would be picked up by git (`.gitignore`) or by the env file
# search (`env.yaml`).
TEMPLATE_RENAMES = {"env.tmpl.yaml": DEFAULT_ENV_FILE, "gitignore": ".gitignore"}

_active: Workspace | None = None  # workspace activated by the CLI; returned by `load_env`


class WorkspaceError(Exception):
    """A workspace could not be found or loaded."""


def _abspath(path: str | pathlib.Path) -> pathlib.Path:
    """Absolute, normalised path without resolving symlinks (unlike `Path.resolve`)."""
    return pathlib.Path(os.path.abspath(path))


def find_env_file(
    env_file: str | pathlib.Path = DEFAULT_ENV_FILE, start: str | pathlib.Path | None = None
) -> pathlib.Path:
    """Locate an env file.

    `env_file` is used as is if it exists relative to `start` (default: current dir) or is absolute. Otherwise, the
    parents of `start` are searched for `<parent>/<env_file>`, so both the default `env.yaml` and a name like
    `env_snellius.yaml` are found from any subdirectory of a workspace.
    """
    env_file = pathlib.Path(env_file)
    start = _abspath(start if start is not None else pathlib.Path.cwd())

    candidates = [env_file] if env_file.is_absolute() else [d / env_file for d in (start, *start.parents)]
    for candidate in candidates:
        if candidate.is_file():
            return _abspath(candidate)

    raise WorkspaceError(
        f"Env file '{env_file}' not found in '{start}' or any of its parents. "
        f"Run this command inside a workspace (create one with `wrf-massive init`) or pass `--env-file`."
    )


def parse_log_level(level: str | int) -> int:
    """Convert a level name like `info` (case-insensitive) or a number to a `logging` level."""
    if isinstance(level, int):
        return level
    levels = logging.getLevelNamesMapping()
    try:
        return levels[str(level).upper()]
    except KeyError:
        valid = ", ".join(name.lower() for name in levels if name != "NOTSET")
        raise WorkspaceError(f"Invalid log_level '{level}'. Choose one of: {valid}.") from None


def configure_logging(level: str | int):
    """Set the root log level and add a stream handler if logging is not configured yet."""
    logging.basicConfig(format=LOG_FORMAT)  # no-op if the root logger already has handlers
    logging.getLogger().setLevel(parse_log_level(level))


class Workspace:
    """A workspace rooted at the directory containing its env file."""

    def __init__(self, env_file: str | pathlib.Path):
        self.env_file = _abspath(env_file)
        self.root = self.env_file.parent  # parent of the file itself, not of a symlink target
        self.env = self._read_env(self.env_file)
        self.log_level = parse_log_level(self.env.get("log_level", DEFAULT_LOG_LEVEL))
        self._pipeline: Pipeline | None = None

    @classmethod
    def discover(cls, env_file: str | pathlib.Path = DEFAULT_ENV_FILE, start: str | pathlib.Path | None = None):
        """Find the env file (see `find_env_file`) and create the workspace from it."""
        return cls(find_env_file(env_file, start=start))

    @staticmethod
    def _read_env(env_file: pathlib.Path) -> Dict[str, Any]:
        env = yaml_to_dict(env_file.read_text())
        if not isinstance(env, dict):
            raise WorkspaceError(f"Env file '{env_file}' must contain a mapping of settings.")
        return env

    def activate(self):
        """Make the workspace current: configure logging, change into the root and make its modules importable.

        Paths in `pipeline.py` and in the env file are thus resolved relative to the root, wherever the CLI was
        started. Idempotent.
        """
        global _active
        configure_logging(self.log_level)
        os.chdir(self.root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        _active = self

    def import_module(self, path: str | pathlib.Path, name: str | None = None) -> ModuleType:
        """Import a Python file of this workspace as module `name` (default: file name without suffix)."""
        path = _abspath(path)
        name = name or path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # register first, so the module can be imported by other workspace modules
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
        return module

    @property
    def job_file(self) -> pathlib.Path | None:
        """SLURM job script of the env file's `job_file` key (relative to the root), or None if not set."""
        value = self.env.get("job_file")
        if not value:
            return None
        path = pathlib.Path(value)
        return path if path.is_absolute() else self.root / path

    @property
    def pipeline(self) -> Pipeline:
        """The pipeline named by the env file's `pipeline` key, loaded from the workspace's `pipeline.py`."""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> Pipeline:
        name = self.env.get("pipeline")
        if not name:
            raise WorkspaceError(f"Env file '{self.env_file}' does not set `pipeline`.")

        module_path = self.root / f"{PIPELINE_MODULE}.py"
        if not module_path.is_file():
            raise WorkspaceError(f"No {module_path.name} found in workspace root '{self.root}'.")

        self.activate()
        try:
            module = self.import_module(module_path, name=PIPELINE_MODULE)
        except Exception as e:
            logger.debug(f"Failed to import {module_path}.", exc_info=True)
            raise WorkspaceError(
                f"Failed to load {module_path} with env file '{self.env_file}': {type(e).__name__}: {e}\n"
                f"Check the settings (e.g. paths) in the env file."
            ) from e

        pipeline = getattr(module, name, None)
        if not isinstance(pipeline, Pipeline):
            available = [k for k, v in vars(module).items() if isinstance(v, Pipeline)]
            raise WorkspaceError(
                f"Pipeline '{name}' (set in '{self.env_file}') not found in {module_path}. "
                f"Available pipelines: {available}"
            )
        logger.debug(f"Loaded pipeline '{name}' from {module_path} using env file '{self.env_file}'.")
        return pipeline


def load_env() -> Dict[str, Any]:
    """Settings of the current workspace's env file, for use in `pipeline.py`.

    Returns the env file the CLI was started with (`--env-file`). Outside the CLI (e.g. in a notebook), the env file
    is taken from `$WRF_MASSIVE_ENV_FILE` or defaults to `env.yaml`, searched from the current directory upwards.
    """
    if _active is not None:
        return _active.env
    return Workspace.discover(os.environ.get(ENV_FILE_ENVVAR, DEFAULT_ENV_FILE)).env


def _template_dir():
    return importlib.resources.files("wrf_massive").joinpath("templates", "workspace")


def template_files() -> Dict[str, str]:
    """Packaged workspace template files, mapped to their name in a new workspace."""
    names = sorted(f.name for f in _template_dir().iterdir() if f.is_file())
    return {name: TEMPLATE_RENAMES.get(name, name) for name in names}


def init_workspace(directory: str | pathlib.Path, force: bool = False) -> List[pathlib.Path]:
    """Copy the workspace template into `directory` (created if missing). Returns the written files.

    Refuses to overwrite existing files unless `force` is set.
    """
    directory = _abspath(directory)
    files = template_files()

    existing = [directory / dst for dst in files.values() if (directory / dst).exists()]
    if existing and not force:
        raise WorkspaceError(
            f"Refusing to overwrite existing files in '{directory}': {[p.name for p in existing]}. "
            f"Use --force to overwrite them."
        )

    directory.mkdir(parents=True, exist_ok=True)
    tmpl_dir = _template_dir()
    written = []
    for src, dst in files.items():
        target = directory / dst
        target.write_bytes(tmpl_dir.joinpath(src).read_bytes())
        written.append(target)
    return written
