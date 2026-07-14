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


def _time_selectors(times: pd.DatetimeIndex) -> Dict[str, List[str]]:
    """CDS-style year/month/day/time selector lists covering `times`.

    CDS expands these into the full year x month x day x time cross-product. That is waste-free only when all
    `times` fall in a single calendar month (see `_month_periods`); across month boundaries the product would
    request spurious dates (a day from one month paired with the other month).
    """
    return {
        "year": sorted({f"{t.year:04d}" for t in times}),
        "month": sorted({f"{t.month:02d}" for t in times}),
        "day": sorted({f"{t.day:02d}" for t in times}),
        "time": sorted({f"{t.hour:02d}:00" for t in times}),
    }


def _month_periods(begin: pd.Timestamp, end: pd.Timestamp, freq: str = "3h") -> Dict[str, pd.DatetimeIndex]:
    """Group the `[begin, end]` timestamps (at `freq` cadence) by calendar month.

    Returns an ordered `{"YYYYMM": times}` mapping. Splitting a CDS pull into one request per month keeps each
    request's year/month/day cross-product waste-free (every day in a group belongs to that group's month),
    which matters for large full-domain pulls (e.g. CERRA) spanning a month boundary.
    """
    times = pd.date_range(start=begin, end=end, freq=freq)
    periods: Dict[str, List[pd.Timestamp]] = {}
    for t in times:
        periods.setdefault(f"{t.year:04d}{t.month:02d}", []).append(t)
    return {key: pd.DatetimeIndex(ts) for key, ts in periods.items()}


class PullCdsStage(Stage):
    """Download forcing data from the Copernicus Climate Data Store (CDS) via the official `cdsapi` client.

    Produces one GRIB file per `CdsRequestSpec` and calendar month, named `<prefix>_<file_suffix>_<YYYYMM>.grb`
    inside `work_dir`. The per-month split keeps each CDS request's year/month/day cross-product waste-free when
    the simulation window straddles a month boundary. Multiple files per source are fine downstream:
    `run_wps.sh` globs `<prefix>*.grb` and `link_grib.csh` concatenates them, so nothing else changes.
    Already-downloaded month files are skipped on re-run, so an interrupted pull resumes where it left off.

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

    def _grib_path(self, s: Simulation, req: CdsRequestSpec, period: str) -> pathlib.Path:
        return self.get_work_dir(s) / f"{self.prefix}_{req.file_suffix}_{period}.grb"

    def _request_path(self, s: Simulation, req: CdsRequestSpec, period: str) -> pathlib.Path:
        # Prefix-scoped like `_grib_path`, so two stages sharing a work_dir (e.g. a CERRA and an ERA5 pull both
        # writing into `1_forcing`) don't collide on their request sidecars when file suffixes overlap.
        return self.get_work_dir(s) / f"cds_request_{self.prefix}_{req.file_suffix}_{period}.yaml"

    def _validate_area(self, s: Simulation):
        # Only area-cropped requests need `Simulation.area`; full-domain requests (use_area=False) do not.
        if not any(req.use_area for req in self.requests):
            return
        if s.area is None:
            raise ValueError("Simulation.area must be set to use PullCdsStage (CDS requests need a lat/lon area).")
        if self.namelist_wps_path is not None:
            domain_bbox = compute_lambert_domain_bbox(self.namelist_wps_path, margin_deg=self.area_margin_deg)
            validate_area_covers_domain(s.area, domain_bbox)

    def _build_request(self, s: Simulation, req: CdsRequestSpec, times: pd.DatetimeIndex) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "product_type": req.product_type,
            "variable": req.variables,
            "data_format": "grib",
            **_time_selectors(times),
        }
        if req.use_area:
            request["area"] = [s.area.north, s.area.west, s.area.south, s.area.east]
        if req.levels is not None:
            request["pressure_level"] = req.levels
        request.update(req.extra_params)
        return request

    def _periods(self, s: Simulation) -> Dict[str, pd.DatetimeIndex]:
        return _month_periods(s.begin_w_warmup, s.end)

    def setup(self, s: Simulation):
        self._validate_area(s)
        for period, times in self._periods(s).items():
            for req in self.requests:
                request = self._build_request(s, req, times)
                request_path = self._request_path(s, req, period)
                request_path.write_text(yaml.dump(request))
                logger.info(f"-> {request_path.name} written.")

    def is_setup(self, s: Simulation) -> bool:
        return all(self._request_path(s, req, period).exists() for period in self._periods(s) for req in self.requests)

    def run(self, s: Simulation):
        import cdsapi

        # wait_until_complete=False makes retrieve() submit the job and return a handle without blocking, so we
        # submit every (month, request) up front and let them queue/process in parallel at CDS, then download.
        # Serialising submit+download instead would make each request sit through the whole CDS queue in turn.
        client = cdsapi.Client(wait_until_complete=False)

        # Phase 1: submit all outstanding requests.
        submitted = []  # (handle, target, label)
        for period in self._periods(s):
            for req in self.requests:
                target = self._grib_path(s, req, period)
                label = f"{req.file_suffix} {period}"
                if target.exists() and target.stat().st_size > 0:
                    logger.info(f"-> {target.name} already present, skipping.")
                    continue
                request = yaml.safe_load(self._request_path(s, req, period).read_text())
                logger.info(f"-> Submitting {req.dataset} ({label}) to CDS...")
                handle = client.retrieve(req.dataset, request)  # no target -> submit only, returns a handle
                submitted.append((handle, target, label))

        # Phase 2: wait for each job (in submission order) and download its result.
        for handle, target, label in submitted:
            logger.info(f"-> Waiting for {label} and downloading...")
            handle.download(str(target))
            logger.info(f"-> Saved to {target}.")

    def is_done(self, s: Simulation) -> bool:
        return all(
            self._grib_path(s, req, period).exists() and self._grib_path(s, req, period).stat().st_size > 0
            for period in self._periods(s)
            for req in self.requests
        )
