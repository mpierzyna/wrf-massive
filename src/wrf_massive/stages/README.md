# Stages of WRF Massive pipeline
1. `forcing`: Retrieve and prepare forcing data for WPS. `PullCerraStage` pulls pre-mirrored GRIB via `rclone`;
   `PullCdsStage` downloads GRIB directly from the Copernicus Climate Data Store via `cdsapi` (see
   `forcing/cds.py`, `forcing/variables.py`, `forcing/geo.py`).
2. `wps`: Run WPS to met_em files
3. `wrf`: Run WRF model
4. `postproc`: Process WRF output data