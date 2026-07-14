from __future__ import annotations

import pathlib
from typing import Any, Callable, Dict, List, Tuple

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


def _time_stamp(t: pd.Timestamp) -> str:
    """`YYYYMMDD_HHMM` key used for per-time grib filenames (matches `_grib_reference_key`)."""
    return f"{t.year:04d}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}"


def _grib_reference_key(header: bytes) -> Tuple[str, int]:
    """Return `(YYYYMMDD_HHMM, total_message_length)` for a GRIB1 or GRIB2 message.

    Reads only the reference date/time, which sits at fixed offsets in the indicator section plus the GRIB1 PDS /
    GRIB2 identification section. This is independent of the product-definition template, so it decodes CERRA's
    local template 4.214 and ERA5's GRIB1 messages alike. `header` must contain at least the first ~40 bytes of
    the message.
    """
    edition = header[7]
    if edition == 2:
        total = int.from_bytes(header[8:16], "big")
        s1 = 16  # identification section immediately follows the 16-byte indicator section
        year = int.from_bytes(header[s1 + 12 : s1 + 14], "big")
        month, day, hour, minute = header[s1 + 14], header[s1 + 15], header[s1 + 16], header[s1 + 17]
    elif edition == 1:
        total = int.from_bytes(header[4:7], "big")
        pds = 8  # product definition section immediately follows the 8-byte indicator section
        year_of_century, month, day = header[pds + 12], header[pds + 13], header[pds + 14]
        hour, minute = header[pds + 15], header[pds + 16]
        century = header[pds + 24]
        year = (century - 1) * 100 + year_of_century
    else:
        raise ValueError(f"Unsupported GRIB edition {edition!r}")
    return f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}", total


def split_grib_by_reference_time(src: pathlib.Path, name_fn: Callable[[str], pathlib.Path]) -> Dict[str, pathlib.Path]:
    """Split a multi-time GRIB file into one file per reference time, streaming byte-for-byte.

    WPS ungrib assigns a single valid time per input file (the first message's, see `ungrib/src/rd_grib2.F`), so a
    monthly multi-time GRIB must be split into per-time files before WPS can decode it. Each message is copied
    verbatim (no decode/re-encode) into the file returned by `name_fn(time_key)`, where `time_key` is
    `YYYYMMDD_HHMM`. Returns the `{time_key: path}` mapping of files written. Handles GRIB1 (ERA5) and GRIB2 (CERRA).
    """
    handles: Dict[str, Any] = {}
    written: Dict[str, pathlib.Path] = {}
    head_len = 40  # enough to cover the reference-time offsets of both editions
    try:
        with open(src, "rb") as f:
            while True:
                header = f.read(head_len)
                if len(header) < 16:
                    break  # end of file
                if header[:4] != b"GRIB":
                    f.seek(-(len(header) - 1), 1)  # resync: step one byte past the failed magic and retry
                    continue
                key, total = _grib_reference_key(header)
                message = header + f.read(total - len(header))
                if key not in handles:
                    path = name_fn(key)
                    handles[key] = open(path, "wb")
                    written[key] = path
                handles[key].write(message)
    finally:
        for handle in handles.values():
            handle.close()
    return written


class PullCdsStage(Stage):
    """Download forcing data from the Copernicus Climate Data Store (CDS) via the official `cdsapi` client.

    Each `CdsRequestSpec` is downloaded as one CDS request per calendar month (keeping the request's
    year/month/day cross-product waste-free when the window straddles a month boundary), then each monthly file is
    split into one GRIB file per valid time, named `<prefix>_<file_suffix>_<YYYYMMDD_HHMM>.grb` inside `work_dir`.
    The per-time split is required because WPS ungrib decodes a single valid time per input file, so a multi-time
    monthly file would collapse onto its first timestep. The per-time files match `run_wps.sh`'s `<prefix>*.grb`
    glob and are concatenated by `link_grib.csh`, so nothing else downstream changes. Already-produced per-time
    files are skipped on re-run, so an interrupted pull resumes where it left off.

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

    def _monthly_path(self, s: Simulation, req: CdsRequestSpec, period: str) -> pathlib.Path:
        # Transient per-month download target; split into per-time files and removed once the split succeeds.
        return self.get_work_dir(s) / f"{self.prefix}_{req.file_suffix}_{period}.grb"

    def _time_path(self, s: Simulation, req: CdsRequestSpec, t: pd.Timestamp) -> pathlib.Path:
        return self.get_work_dir(s) / f"{self.prefix}_{req.file_suffix}_{_time_stamp(t)}.grb"

    def _request_path(self, s: Simulation, req: CdsRequestSpec, period: str) -> pathlib.Path:
        # Prefix-scoped like the grib paths, so two stages sharing a work_dir (e.g. a CERRA and an ERA5 pull both
        # writing into `1_forcing`) don't collide on their request sidecars when file suffixes overlap.
        return self.get_work_dir(s) / f"cds_request_{self.prefix}_{req.file_suffix}_{period}.yaml"

    def _times_complete(self, s: Simulation, req: CdsRequestSpec, times: pd.DatetimeIndex) -> bool:
        return all(self._time_path(s, req, t).exists() and self._time_path(s, req, t).stat().st_size > 0 for t in times)

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

    def _split_monthly(self, s: Simulation, req: CdsRequestSpec, monthly: pathlib.Path):
        # Split the downloaded monthly file into per-time files, then drop the monthly aggregate so it is not also
        # picked up by run_wps.sh's `<prefix>*.grb` glob (which would double-feed ungrib).
        name_fn = lambda key: self.get_work_dir(s) / f"{self.prefix}_{req.file_suffix}_{key}.grb"
        written = split_grib_by_reference_time(monthly, name_fn)
        logger.info(f"-> Split {monthly.name} into {len(written)} per-time files.")
        monthly.unlink()

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

        # Phase 1: submit outstanding requests. `pending_split` also carries any month already downloaded (but not
        # yet split) from a previous interrupted run, so we resume at the split step without re-downloading.
        submitted = []  # (handle, monthly, req, label)
        pending_split = []  # (monthly, req, label)
        for period, times in self._periods(s).items():
            for req in self.requests:
                label = f"{req.file_suffix} {period}"
                if self._times_complete(s, req, times):
                    logger.info(f"-> {self.prefix} {label} per-time files already present, skipping.")
                    continue
                monthly = self._monthly_path(s, req, period)
                if monthly.exists() and monthly.stat().st_size > 0:
                    pending_split.append((monthly, req, label))  # downloaded earlier, only the split is outstanding
                    continue
                request = yaml.safe_load(self._request_path(s, req, period).read_text())
                logger.info(f"-> Submitting {req.dataset} ({label}) to CDS...")
                handle = client.retrieve(req.dataset, request)  # no target -> submit only, returns a handle
                submitted.append((handle, monthly, req, label))

        # Phase 2: wait for each job (in submission order) and download its result.
        for handle, monthly, req, label in submitted:
            logger.info(f"-> Waiting for {label} and downloading...")
            handle.download(str(monthly))
            logger.info(f"-> Saved to {monthly}.")
            pending_split.append((monthly, req, label))

        # Phase 3: split every downloaded month into the per-time files WPS ungrib needs.
        for monthly, req, label in pending_split:
            self._split_monthly(s, req, monthly)

    def is_done(self, s: Simulation) -> bool:
        return all(
            self._times_complete(s, req, times) for times in self._periods(s).values() for req in self.requests
        )
