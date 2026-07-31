"""Logic defining stages and pipelines acting together"""

from __future__ import annotations

import abc
import contextlib
import datetime
import os
import pathlib
from typing import Annotated, Dict, Iterator, List, Union

import pydantic

from wrf_massive.config import BaseYAMLConfig, yaml_to_dict
from wrf_massive.log import get_logger
from wrf_massive.tmp_dir import setup_tmp_work_dir, teardown_tmp_work_dir

logger = get_logger()


def _parse_datetime(v: str) -> datetime.datetime:
    """Parse a datetime from an ISO formatted string."""
    if isinstance(v, datetime.datetime):
        return v
    return datetime.datetime.fromisoformat(v)


def _ensure_path(v: str | pathlib.Path) -> pathlib.Path:
    """Convert the given value to a pathlib.Path."""
    return pathlib.Path(v)


def _ensure_path_exists(v: str | pathlib.Path) -> pathlib.Path:
    """Ensure the given path exists, otherwise raise ValueError."""
    p = pathlib.Path(v)
    if not p.exists():
        raise ValueError(f"Path {p} does not exist!")
    return p


def _mkdir_if_not_exists(p: str | pathlib.Path) -> pathlib.Path:
    """Create the given directory if it does not exist."""
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


TPath = Annotated[Union[pathlib.Path, str], pydantic.BeforeValidator(_ensure_path)]
TPathExists = Annotated[Union[pathlib.Path, str], pydantic.BeforeValidator(_ensure_path_exists)]
TPathMkdir = Annotated[Union[pathlib.Path, str], pydantic.BeforeValidator(_mkdir_if_not_exists)]


class Resources(pydantic.BaseModel):
    """Resources required by a stage, e.g. for job submission.

    Note
    ----
    - Python processes typically need 1 task, n CPUs
    - MPI programs typically need n tasks, 1 CPU per task
    """

    n_tasks: int  # number of tasks
    cpus_per_task: int  # number of CPUs per task
    mem_per_cpu: str  # memory per CPU (e.g. "4G", "800M")
    walltime: datetime.timedelta | None = None  # maximum walltime
    max_concurrent: int | None = pydantic.Field(default=None, gt=0)
    # If set, caps the number of concurrently RUNNING jobs for this stage at this value,
    # independent of available CPU/core headroom. Implemented via SLURM's
    # `--dependency=singleton` (max concurrency 1 per unique --job-name/user), extended to N
    # by hashing each sim_dir into one of N lanes sharing an identical --job-name (see
    # `_submit_stage_slurm` in cli.py).

    @property
    def cpus_total(self) -> int:
        return self.n_tasks * self.cpus_per_task


class Stage(pydantic.BaseModel, abc.ABC):

    work_dir: TPath  # if relative, relative to sim_dir
    resources: Resources | None = None  # resources required by this stage

    # Run this stage's work dir on fast temporary storage (ramdisk, node-local or shared scratch).
    # The work dir is moved to `<tmp_work_root>/<sim.name>/<work_dir>` and the original path becomes
    # a symlink to it, so stage implementations need no changes at all: `get_work_dir()` keeps
    # returning the same path and the filesystem does the redirection. See `Stage.tmp_work_dir`.
    tmp_work_root: TPathExists | None = None  # None disables the mechanism entirely
    tmp_teardown_globs: List[str] | None = None  # None moves the whole dir back, else only these globs
    tmp_skip_teardown: bool = False  # leave results in tmp even on success (debugging, shared scratch)

    _tmp_depth: int = pydantic.PrivateAttr(default=0)  # re-entrancy guard for `tmp_work_dir`

    @property
    def name(self) -> str:
        """Name of the class is name of the stage"""
        return self.__class__.__name__

    @abc.abstractmethod
    def is_setup(self, s: Simulation) -> bool: ...

    @abc.abstractmethod
    def setup(self, s: Simulation): ...

    @abc.abstractmethod
    def is_done(self, s: Simulation) -> bool: ...

    @abc.abstractmethod
    def run(self, s: Simulation): ...

    def get_work_dir(self, s: Simulation, create: bool = True) -> pathlib.Path:
        """Get working directory, either relative to sim_dir or forwarding absolute path."""
        work_dir = self.work_dir
        if work_dir.is_absolute():
            return work_dir
        work_dir = s.sim_dir / work_dir
        if create:
            if work_dir.is_symlink() and not work_dir.exists():
                # Dangling symlink into a tmp root that was wiped since the last run. `exists()`
                # follows symlinks, so this is exactly the dangling case. Without this, the mkdir
                # below would raise FileExistsError and every run would be stuck.
                logger.warning(f"Work dir '{work_dir}' is a dangling symlink (-> {os.readlink(work_dir)}). Removing.")
                work_dir.unlink()
            work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _check_tmp_work_dir_allowed(self, s: Simulation) -> None:
        """Raise if this stage's work dir must never be relocated to temporary storage."""
        if self.work_dir.is_absolute():
            # `tmp_root / s.name / <absolute>` collapses to the absolute path itself -> self-symlink
            raise ValueError(
                f"Stage '{self.name}' has an absolute work_dir ('{self.work_dir}'), "
                "which cannot be run in a tmp work dir."
            )
        work_dir = self.get_work_dir(s, create=False).absolute()
        sim_dir = s.sim_dir.absolute()
        if work_dir == sim_dir or sim_dir.is_relative_to(work_dir):
            # e.g. MarkDone(work_dir=".") -> would move the ENTIRE simulation dir to tmp storage
            raise ValueError(
                f"Stage '{self.name}' has work_dir '{self.work_dir}', which is (or contains) the simulation "
                f"dir '{sim_dir}'. Refusing to move it to tmp storage."
            )

    def tmp_work_dir_allowed(self, s: Simulation) -> bool:
        """Non-raising variant of `_check_tmp_work_dir_allowed`."""
        try:
            self._check_tmp_work_dir_allowed(s)
        except ValueError:
            return False
        return True

    @contextlib.contextmanager
    def tmp_work_dir(
        self,
        s: Simulation,
        *,
        root: pathlib.Path | None = None,
        teardown_globs: List[str] | None = None,
        skip_teardown: bool | None = None,
    ) -> Iterator[pathlib.Path | None]:
        """Run this stage's work dir on fast temporary storage for the duration of the block.

        No-op unless a tmp root is configured (either the `tmp_work_root` field or the `root` argument).
        The keyword arguments let a parent stage (e.g. `StageArray`) drive its substages without mutating them.

        On success the results are moved back to the simulation dir. If the block raises, the work dir is
        left symlinked into tmp and the exception is re-raised: a re-run then resumes in place, because
        `setup_tmp_work_dir` is idempotent.
        """
        root = root if root is not None else self.tmp_work_root
        if root is None or self._tmp_depth > 0:
            # Mechanism disabled, or we are already inside the context -> do nothing
            yield None
            return

        self._check_tmp_work_dir_allowed(s)
        work_dir_tmp = setup_tmp_work_dir(tmp_root=root, s=s, stage=self)
        self._tmp_depth += 1
        try:
            yield work_dir_tmp
        except BaseException:
            self._tmp_depth -= 1
            logger.error(
                f"Stage '{self.name}' failed. Work dir is LEFT in '{work_dir_tmp}' "
                f"(still symlinked from '{self.get_work_dir(s, create=False)}'). A re-run resumes in place."
            )
            raise
        else:
            # Decrement before tearing down so teardown can call back into `get_work_dir`/this context
            self._tmp_depth -= 1
            if skip_teardown if skip_teardown is not None else self.tmp_skip_teardown:
                logger.info(f"Teardown disabled for '{self.name}'. Work dir stays in '{work_dir_tmp}'.")
                return
            teardown_tmp_work_dir(
                s=s,
                stage=self,
                move_globs=teardown_globs if teardown_globs is not None else self.tmp_teardown_globs,
            )


class BBox(pydantic.BaseModel):
    """Geographic lat/lon bounding box (degrees), e.g. for CDS/ARCO `area` requests."""

    north: float
    west: float
    south: float
    east: float


class Simulation(BaseYAMLConfig):
    """Container and config object of a SINGLE WRF simulation."""

    # Begin and end of simulation EXCLUDING warmup
    begin: Annotated[datetime.datetime | str, pydantic.BeforeValidator(_parse_datetime)]
    end: Annotated[datetime.datetime | str, pydantic.BeforeValidator(_parse_datetime)]
    warmup_h: int = 12

    # Simulation settings to be rendered in namelist.input
    settings: Dict[str, str]

    # Simulation directory
    sim_dir: TPath

    # Lat/lon bounding box forcing-data downloads (e.g. CDS `area`) should cover. Must fully contain the WRF
    # domain (see stages/forcing/geo.py::validate_area_covers_domain).
    area: BBox | None = None

    @property
    def name(self) -> str:
        """Use sim directory name as simulation name."""
        return self.sim_dir.name

    @property
    def begin_w_warmup(self) -> datetime.datetime:
        """Begin of simulation INCLUDING warmup period."""
        return self.begin - datetime.timedelta(hours=self.warmup_h)

    def to_disk(self, root: str | pathlib.Path = "."):
        """Save simulation config to disk `sim_dir`."""
        # Make sim dir inside specified root
        root = pathlib.Path(root)
        p = root / self.sim_dir
        p.mkdir(parents=True, exist_ok=True)
        # Don't save sim_dir because it is directory containing the config itself
        (p / "simulation.yaml").write_text(self.model_dump_yaml(exclude={"sim_dir": ...}))

    @classmethod
    def from_disk(cls, sim_dir: str | pathlib.Path) -> Simulation:
        """Load simulation config from disk `sim_dir`."""
        sim_dir = pathlib.Path(sim_dir)
        if not sim_dir.exists():
            raise ValueError(f"Simulation directory {sim_dir} does not exist!")
        sim_yaml = (sim_dir / "simulation.yaml").read_text()
        sim_dict = yaml_to_dict(sim_yaml)
        sim_dict["sim_dir"] = sim_dir  # add sim_dir back to dict
        return cls(**sim_dict)


class Pipeline:
    """A pipeline is a sequence of stages to be executed in order."""

    def __init__(self, **stages: Stage):
        self.stages: Dict[str, Stage] = {}
        self.add_stages(**stages)

    @property
    def stage_names(self) -> List[str]:
        """Names of all stages in the pipeline."""
        return list(self.stages.keys())

    def run_stage(self, s: Simulation, name: str, force_setup: bool = False, force_run: bool = False):
        """Run a single stage by name."""
        # Get stage object
        if name not in self.stages:
            raise ValueError(f"Stage with name {name} not found in pipeline!")
        stage = self.stages[name]

        # mostly for debugging or to avoid downloading large files again
        # (create=False: don't materialise a dir for a stage we may skip, and don't pre-create one
        #  that the tmp setup below would then have to move again)
        if (stage.get_work_dir(s, create=False) / ".done").exists():
            logger.info(f"Stage '{name}' is marked to be skipped (found .done file). Skipping.")
            return

        # Nothing to do: same outcome as the block below, but skips moving the work dir to tmp
        # storage and back for an already-complete simulation. Safe to check outside the tmp
        # context: a symlink left over from an earlier failure is still live, so these checks
        # resolve to the same files either way.
        if not force_setup and not force_run and stage.is_setup(s) and stage.is_done(s):
            logger.info(f"Stage '{name}' already setup and done. Skipping.")
            return

        # Optionally relocate the stage work dir to fast temporary storage. This is a no-op unless
        # the stage has `tmp_work_root` set. Note that is_setup/is_done below are evaluated INSIDE
        # the context, i.e. they inspect the tmp dir through the symlink.
        with stage.tmp_work_dir(s):
            # Setup
            if not stage.is_setup(s) or force_setup:
                logger.info(f"Setting up '{name}'...")
                stage.setup(s)
            else:
                logger.info(f"Stage '{name}' already setup, skipping setup.")

            # Run
            if stage.is_done(s) and not force_run:
                logger.info(f"Stage '{name}' already done, skipping run.")
                return  # normal exit -> the context still tears the tmp dir down
            logger.info(f"Running stage '{name}'...")
            stage.run(s)

    def run(
        self,
        s: Simulation,
        stages: List[str] | str | None = None,
        force_setup: bool = False,
        force_run: bool = False,
    ):
        """Run the full pipeline, or a subset of stages if `stage` is given."""
        logger.info(f"Processing simulation {s.begin} -> {s.end} (warmup: {s.warmup_h}h)")

        if (s.sim_dir / ".done").exists():
            logger.info(f"Simulation '{s.sim_dir}' is marked as done (found .done file)! Skipping.")
            return

        # Prepare stages input
        if isinstance(stages, str):
            stages = [stages]
        elif stages is None:
            stages = self.stage_names  # by default, select all stages for running

        # Run stages
        n = len(stages)
        for i, name in enumerate(stages):
            logger.info(f"Entering stage #{i+1}/{n}: {name}")
            self.run_stage(s=s, name=name, force_setup=force_setup, force_run=force_run)

    def add_stages(self, **stages: Stage):
        """Add one or more stages to the pipeline."""
        for name, stage in stages.items():
            if name in self.stages:
                raise ValueError(f"Stage with name {name} already exists in pipeline!")
            self.stages[name] = stage

    def __getitem__(self, name) -> Stage:
        return self.stages[name]
