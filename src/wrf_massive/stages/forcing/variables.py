"""CDS request variable/level lists for ERA5 and CERRA, derived from what ungrib/WPS actually reads.

The authoritative source for "what does WPS need" is the Vtable files linked in `stages/wps/run_wps.sh`:
- `stages/wps/Vtable.CERRA` (ships in this repo)
- `Vtable.ERA-interim.pl` (ships with the WPS submodule, `ungrib/Variable_Tables/`, used unmodified for ERA5)

ERA5 variable names below follow CDS's long-standing, stable `reanalysis-era5-pressure-levels` /
`reanalysis-era5-single-levels` catalogue naming and are used with high confidence.

CERRA names could NOT be verified against the live CDS catalogue from this environment (outbound access to
cds.climate.copernicus.eu is blocked here). In particular, CERRA's `reanalysis-cerra-single-levels` dataset may
expose 10m wind as speed/direction (`10m_wind_speed`/`10m_wind_direction`) rather than u/v components -- while
`Vtable.CERRA` expects u/v components (GRIB1 params 165/166, `UU`/`VV`). If that's the case for your CDS account,
either request the component-wind variant if/when the catalogue offers one, or add a speed+direction -> u/v
conversion step before ungrib. Double-check every CERRA variable/level string below against the CDS dataset's own
"Download data" request-builder before relying on it.
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
# NOTE: unverified against the live CDS catalogue -- see module docstring.

CERRA_PRESSURE_LEVEL_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

CERRA_SINGLE_LEVEL_VARIABLES = [
    "10m_wind_speed",  # VERIFY: Vtable.CERRA wants u/v components (params 165/166), not speed/direction
    "10m_wind_direction",  # VERIFY: see above
    "2m_temperature",
    "2m_relative_humidity",
    "surface_pressure",
    "mean_sea_level_pressure",
    "skin_temperature",
    "snow_depth",
    "land_sea_mask",
    "orography",
]
