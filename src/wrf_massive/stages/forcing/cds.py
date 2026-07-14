from __future__ import annotations

import pathlib
from typing import Any, Dict, List

import pandas as pd
import pydantic
import yaml

from wrf_massive.base import Simulation, Stage, TPathExists
from wrf_massive.log import get_logger
from wrf_massive.stages.forcing.geo import compute_lambert_domain_bbox, validate_area_covers_domain

logger = get_logger("stages.cds")


class CdsRequestSpec(pydantic.BaseModel):
    """One CDS API request: a dataset plus the variables (and optional pressure levels) to pull from it."""

    dataset: str  # CDS dataset name, e.g. "reanalysis-era5-pressure-levels"
    variables: List[str]
    levels: List[str] | None = None  # pressure levels; required for pressure-level datasets
    file_suffix: str  # combined with `PullCdsStage.prefix` for the output grib filename, e.g. "PRES", "SFC"
    product_type: List[str] = ["reanalysis"]  # ERA5 uses "reanalysis"; CERRA uses "analysis"/"forecast"
    use_area: bool = True  # add a lat/lon `area` crop to the request. Disable for datasets that do not support
    # geographic subsetting (e.g. CERRA's projected grid) -- the full native domain is downloaded and WPS crops it.
    extra_params: Dict[str, Any] = {}  # dataset-specific overrides/additions (e.g. CERRA's `data_type`)


def _build_time_selectors(begin: pd.Timestamp, end: pd.Timestamp, freq: str = "3h") -> Dict[str, List[str]]:
    """Build CDS-style year/month/day/time selector lists covering `[begin, end]` at `freq` cadence.

    Note this is a cross-product (all years x all months x all days x all times seen in the range), so it may
    request a superset of timestamps outside `[begin, end]` for multi-month/-year spans. Harmless for WPS/WRF
    (unused timestamps are simply ignored), just not maximally frugal on download size.
    """
    times = pd.date_range(start=begin, end=end, freq=freq)
    return {
        "year": sorted({f"{t.year:04d}" for t in times}),
        "month": sorted({f"{t.month:02d}" for t in times}),
        "day": sorted({f"{t.day:02d}" for t in times}),
        "time": sorted({f"{t.hour:02d}:00" for t in times}),
    }


class PullCdsStage(Stage):
    """Download forcing data from the Copernicus Climate Data Store (CDS) via the official `cdsapi` client.

    Produces one GRIB file per `CdsRequestSpec` in `requests`, named `<prefix>_<file_suffix>.grb` inside
    `work_dir` -- this already matches `run_wps.sh`'s existing `find $FORCING_DIR -name '<prefix>*.grb'` +
    ungrib.exe flow, so no changes are needed downstream of this stage.

    CDS credentials are resolved by `cdsapi.Client()` itself, using its own standard lookup (`~/.cdsapirc`, or
    the `CDSAPI_URL`/`CDSAPI_KEY` environment variables) -- nothing extra to configure here.

    Per-request knobs on `CdsRequestSpec` cover the differences between datasets: `product_type` ("reanalysis"
    for ERA5, "analysis" for CERRA), `use_area` (ERA5 supports lat/lon cropping; CERRA's projected grid is
    downloaded in full and cropped by WPS), and `extra_params` for anything else a dataset requires.
    """

    prefix: str  # WPS ungrib prefix, must match Vtable/run_wps.sh/fg_name conventions (e.g. "CERRA", "ERA5")
    requests: List[CdsRequestSpec]
    namelist_wps_path: TPathExists | None = None  # if given, validate Simulation.area covers the WRF domain
    area_margin_deg: float = 1.0

    def _grib_path(self, s: Simulation, req: CdsRequestSpec) -> pathlib.Path:
        return self.get_work_dir(s) / f"{self.prefix}_{req.file_suffix}.grb"

    def _request_path(self, s: Simulation, req: CdsRequestSpec) -> pathlib.Path:
        # Prefix-scoped like `_grib_path`, so two stages sharing a work_dir (e.g. a CERRA and an ERA5 pull both
        # writing into `1_forcing`) don't collide on their request sidecars when file suffixes overlap.
        return self.get_work_dir(s) / f"cds_request_{self.prefix}_{req.file_suffix}.yaml"

    def _validate_area(self, s: Simulation):
        # Only area-cropped requests need `Simulation.area`; full-domain requests (use_area=False) do not.
        if not any(req.use_area for req in self.requests):
            return
        if s.area is None:
            raise ValueError("Simulation.area must be set to use PullCdsStage (CDS requests need a lat/lon area).")
        if self.namelist_wps_path is not None:
            domain_bbox = compute_lambert_domain_bbox(self.namelist_wps_path, margin_deg=self.area_margin_deg)
            validate_area_covers_domain(s.area, domain_bbox)

    def _build_request(self, s: Simulation, req: CdsRequestSpec) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "product_type": req.product_type,
            "variable": req.variables,
            "data_format": "grib",
            **_build_time_selectors(s.begin_w_warmup, s.end),
        }
        if req.use_area:
            request["area"] = [s.area.north, s.area.west, s.area.south, s.area.east]
        if req.levels is not None:
            request["pressure_level"] = req.levels
        request.update(req.extra_params)
        return request

    def setup(self, s: Simulation):
        self._validate_area(s)
        for req in self.requests:
            request = self._build_request(s, req)
            request_path = self._request_path(s, req)
            request_path.write_text(yaml.dump(request))
            logger.info(f"-> {request_path.name} written.")

    def is_setup(self, s: Simulation) -> bool:
        return all(self._request_path(s, req).exists() for req in self.requests)

    def run(self, s: Simulation):
        import cdsapi

        client = cdsapi.Client()
        for req in self.requests:
            request = yaml.safe_load(self._request_path(s, req).read_text())
            target = self._grib_path(s, req)
            logger.info(f"-> Requesting {req.dataset} ({req.file_suffix}) from CDS...")
            client.retrieve(req.dataset, request, str(target))
            logger.info(f"-> Saved to {target}.")

    def is_done(self, s: Simulation) -> bool:
        return all(
            self._grib_path(s, req).exists() and self._grib_path(s, req).stat().st_size > 0 for req in self.requests
        )
