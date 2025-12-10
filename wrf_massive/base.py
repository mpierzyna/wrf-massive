"""Logic defining stages and pipelines acting together"""

from __future__ import annotations

import abc
import datetime
import pathlib
from typing import Dict, Annotated, Union
from typing import List

import pydantic

from research_tools.misc_tools.yaml_config import BaseYAMLConfig, yaml_to_dict
from wrf_massive.log import get_logger

logger = get_logger(__name__)


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

    @property
    def cpus_total(self) -> int:
        return self.n_tasks * self.cpus_per_task


class Stage(pydantic.BaseModel, abc.ABC):

    work_dir: TPath  # if relative, relative to sim_dir
    resources: Resources | None = None  # resources required by this stage

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
            work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir


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
        if (stage.get_work_dir(s) / ".done").exists():
            logger.info(f"Stage '{name}' is marked to be skipped (found .done file). Skipping.")
            return

        # Setup
        if not stage.is_setup(s) or force_setup:
            logger.info(f"Setting up '{name}'...")
            stage.setup(s)
        else:
            logger.info(f"Stage '{name}' already setup, skipping setup.")

        # Run
        if stage.is_done(s) and not force_run:
            logger.info(f"Stage '{name}' already done, skipping run.")
            return
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
