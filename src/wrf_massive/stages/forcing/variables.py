"""CDS request variable/level lists for ERA5 and CERRA, derived from what ungrib/WPS actually reads.

The authoritative source for "what does WPS need" is the Vtable files linked in `stages/wps/run_wps.sh`:
- `stages/wps/Vtable.CERRA` (ships in this repo)
- `Vtable.ERA-interim.pl` (ships with the WPS submodule, `ungrib/Variable_Tables/`, used unmodified for ERA5)

The ERA5 and CERRA lists below match what `Vtable.CERRA` / `Vtable.ERA-interim.pl` consume and follow the
CDS catalogue naming for their respective datasets.

Note on CERRA 10m wind: `reanalysis-cerra-single-levels` provides 10m wind as speed + direction
(`10m_wind_speed`/`10m_wind_direction`), not as u/v components, whereas `Vtable.CERRA` needs u/v (params
165/166) and ungrib/metgrid cannot rotate speed/dir -> u/v. `CERRA_SINGLE_LEVEL_VARIABLES` therefore omits
10m wind; source 10m u/v from ERA5 instead (`ERA5_SINGLE_LEVEL_VARIABLES`), which metgrid merges, or add a
speed/dir -> u/v conversion step if native-resolution CERRA 10m wind is required.
"""

from __future__ import annotations

# ERA5 -----------------------------------------------------------------------------------------------------------

ERA5_PRESSURE_LEVELS = [
    "1",
    "2",
    "3",
    "5",
    "7",
    "10",
    "20",
    "30",
    "50",
    "70",
    "100",
    "125",
    "150",
    "175",
    "200",
    "225",
    "250",
    "300",
    "350",
    "400",
    "450",
    "500",
    "550",
    "600",
    "650",
    "700",
    "750",
    "775",
    "800",
    "825",
    "850",
    "875",
    "900",
    "925",
    "950",
    "975",
    "1000",
]

ERA5_PRESSURE_LEVEL_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

ERA5_SINGLE_LEVEL_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "mean_sea_level_pressure",
    "skin_temperature",
    "sea_ice_cover",
    "sea_surface_temperature",
    "snow_depth",  # water-equivalent depth ("SNOW_EC" in Vtable.ERA-interim.pl)
    "snow_density",  # ("SNOW_DEN"); metgrid derives physical SNOWH from these two
    "land_sea_mask",
    "geopotential",  # static/time-invariant; used for orography ("SOILGEO"/"SOILHGT")
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]

# CERRA ----------------------------------------------------------------------------------------------------------

# Pressure levels (hPa) available in `reanalysis-cerra-pressure-levels`.
CERRA_PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
    "100", "150", "200", "250", "300", "400", "500", "600", "700", "750",
    "800", "825", "850", "875", "900", "925", "950", "975", "1000",
]  # fmt: skip

CERRA_PRESSURE_LEVEL_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

# 10m wind is intentionally excluded: CERRA archives it as speed/direction, which ungrib cannot turn into the
# u/v components `Vtable.CERRA` expects -- source 10m u/v from ERA5 instead (see module docstring).
CERRA_SINGLE_LEVEL_VARIABLES = [
    "2m_temperature",
    "2m_relative_humidity",
    "surface_pressure",
    "mean_sea_level_pressure",
    "skin_temperature",
    "land_sea_mask",
    "orography",
    "snow_depth",  # physical snow depth (m); "SNOWH" in Vtable.CERRA
    "snow_depth_water_equivalent",  # (kg m-2); "SNOW" in Vtable.CERRA
]
