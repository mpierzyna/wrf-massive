"""Pipelines each simulation of this workspace goes through.

A pipeline is an ordered set of named stages. This one
1. downloads CERRA (atmosphere, surface) and ERA5 (soil, 10 m wind) forcing data from the Copernicus Climate Data
   Store (CDS), which needs CDS credentials (`~/.cdsapirc` or `CDSAPI_URL`/`CDSAPI_KEY`),
2. runs WPS on the forcing data,
3. runs WRF and
4. marks the simulation as done.

The env file (`env.yaml`) selects the pipeline (`pipeline: p_default`) and provides the machine-specific settings (`load_env()`).
Relative paths are relative to the workspace root.
"""

from wrf_massive.base import Pipeline, Resources
from wrf_massive.stages.forcing import CdsRequestSpec, PullCdsStage
from wrf_massive.stages.forcing.variables import (
    CERRA_PRESSURE_LEVEL_VARIABLES,
    CERRA_PRESSURE_LEVELS,
    CERRA_SINGLE_LEVEL_VARIABLES,
)
from wrf_massive.stages.misc import MarkDone
from wrf_massive.stages.wps import WPSStage
from wrf_massive.stages.wrf import WRFStage
from wrf_massive.workspace import load_env

env = load_env()

FORCING_DIR = "1_forcing"

# CERRA provides the atmosphere and surface. Its projected grid is downloaded in full (no `area` crop) and cropped
# by WPS. Alternative without CDS: `PullCerraStage` copies pre-mirrored CERRA files from an rclone remote.
_cerra = PullCdsStage(
    work_dir=FORCING_DIR,
    prefix="CERRA",
    requests=[
        CdsRequestSpec(
            dataset="reanalysis-cerra-pressure-levels",
            variables=CERRA_PRESSURE_LEVEL_VARIABLES,
            levels=CERRA_PRESSURE_LEVELS,
            file_suffix="PRES",
            product_type=["analysis"],
            use_area=False,
            extra_params={"data_type": ["reanalysis"]},
        ),
        CdsRequestSpec(
            dataset="reanalysis-cerra-single-levels",
            variables=CERRA_SINGLE_LEVEL_VARIABLES,
            file_suffix="SFC",
            product_type=["analysis"],
            use_area=False,
            extra_params={"data_type": ["reanalysis"], "level_type": ["surface_or_atmosphere"]},
        ),
    ],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="2G", max_concurrent=2),
)

# ERA5 fills what CERRA lacks: soil layers and 10 m wind as u/v components. Cropped to `Simulation.area`, which is
# checked to cover the WRF domain in namelist.tmpl.wps.
_era5 = PullCdsStage(
    work_dir=FORCING_DIR,
    prefix="ERA5",
    namelist_wps_path="namelist.tmpl.wps",
    requests=[
        CdsRequestSpec(
            dataset="reanalysis-era5-single-levels",
            variables=[
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "land_sea_mask",
                "soil_temperature_level_1",
                "soil_temperature_level_2",
                "soil_temperature_level_3",
                "soil_temperature_level_4",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
            ],
            file_suffix="SFC",
        ),
    ],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="2G", max_concurrent=2),
)

_wps = WPSStage(
    work_dir="2_wps",
    forcing_dir=FORCING_DIR,
    namelist_tmpl_path="namelist.tmpl.wps",
    **env["wps"],
    resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="2G"),  # serial WPS
)

_wrf = WRFStage(
    work_dir="3_wrf",
    met_em_dir=_wps.work_dir,
    namelist_tmpl_path="namelist.tmpl.input",
    myoutfields_path="myoutfields.txt",
    **env["wrf"],
    resources=Resources(n_tasks=4, cpus_per_task=1, mem_per_cpu="2G"),
)

# Mark whole simulation dir as done when all stages complete
_sim_done = MarkDone(work_dir=".", resources=Resources(n_tasks=1, cpus_per_task=1, mem_per_cpu="100M"))

p_default = Pipeline(
    cerra=_cerra,
    era5=_era5,
    wps=_wps,
    wrf=_wrf,
    sim_done=_sim_done,
)
