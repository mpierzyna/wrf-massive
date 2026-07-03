"""Setup the pipeline each WRF simulation will go through.

Typically, a pipeline has the following stages:
- Retrieve external forcing data (GFS, ERA5, CERRA) for the area and time of the simulation
- Process the forcing data with WPS
- Use the WPS output to run WRF
- Postprocess the WRF output depending on scientific objectives.
"""

import logging
import pathlib

from simulations import sim_test
from wrf_massive.base import Pipeline, Resources, Stage
from wrf_massive.config import yaml_to_dict
from wrf_massive.stages.forcing import CdsRequestSpec, PullCdsStage, PullCerraStage
from wrf_massive.stages.forcing.variables import (
    ERA5_PRESSURE_LEVEL_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SINGLE_LEVEL_VARIABLES,
)
from wrf_massive.stages.misc import MarkDone
from wrf_massive.stages.wps import WPSStage
from wrf_massive.stages.wrf import WRFStage


def update_resources(stage: Stage, **resources) -> Stage:
    """Helper to update n_tasks of a stage. Returns deep copy."""
    import copy

    stage = copy.deepcopy(stage)
    stage.resources = stage.resources.model_copy(update=resources)
    return stage


# Load host-specific environment settings
env = yaml_to_dict(pathlib.Path("env.yaml").read_text())

# Setup stages
_cerra = PullCerraStage(
    work_dir="1_forcing",
    remote_path="tudelft:staff-umbrella/HBaki/CERRA",
    remote_flist_path="CERRA_files.txt",
    n_transfers=4,
    resources=Resources(n_tasks=1, cpus_per_task=4, mem_per_cpu="1G"),
)

# Alternative to `_cerra`: download ERA5 directly from CDS instead of an rclone mirror (see
# stages/forcing/cds.py). CERRA works the same way, just point `dataset`/`variables` at the CERRA
# equivalents in `stages/forcing/variables.py` -- double-check those against your CDS account first,
# see the module docstring.
_era5_cds = PullCdsStage(
    work_dir="1_forcing",
    prefix="ERA5",
    namelist_wps_path="namelist.tmpl.wps",  # validates Simulation.area covers the WRF domain
    requests=[
        CdsRequestSpec(
            dataset="reanalysis-era5-pressure-levels",
            variables=ERA5_PRESSURE_LEVEL_VARIABLES,
            levels=ERA5_PRESSURE_LEVELS,
            file_suffix="PRES",
        ),
        CdsRequestSpec(
            dataset="reanalysis-era5-single-levels",
            variables=ERA5_SINGLE_LEVEL_VARIABLES,
            file_suffix="SFC",
        ),
    ],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),
)

_wps = WPSStage(
    work_dir="2_wps",
    forcing_dir=_cerra.work_dir,  # 1_forcing
    namelist_tmpl_path="namelist.tmpl.wps",
    **env["wps"],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="1G"),  # serial WPS
)

_wrf = WRFStage(
    work_dir="3_wrf",
    met_em_dir=_wps.work_dir,  # 2_wps
    namelist_tmpl_path="namelist.tmpl.input",
    myoutfields_path="myoutfields.txt",
    **env["wrf"],
    resources=Resources(n_tasks=4, cpus_per_task=1, mem_per_cpu="1G"),
)
_sim_done = MarkDone(work_dir=".")  # mark whole simulation dir as done when all stages complete

# Assemble default pipeline
p_default = Pipeline(
    cerra=_cerra,
    wps=_wps,
    wrf=_wrf,
    sim_done=_sim_done,
)

# Alternative: forcing data downloaded from CDS instead of pulled from an rclone mirror. Requires
# `Simulation.area` to be set (see simulations.py) and CDS credentials (`~/.cdsapirc` or
# `CDSAPI_URL`/`CDSAPI_KEY`).
p_cds = Pipeline(
    era5_cds=_era5_cds,
    wps=_wps,
    wrf=_wrf,
    sim_done=_sim_done,
)


if env["machine"] == "hpc":
    # HPC example.
    # Assume WPS was run externally and only WPS output is copied to HPC -> Forcing and WPS stage not needed.
    # HPC has more resources, so we increase the allocations for WRF.
    p_hpc = Pipeline(
        wrf=update_resources(_wrf, n_tasks=32),
        sim_done=_sim_done,
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    p_default.run(sim_test)
