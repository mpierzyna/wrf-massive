from __future__ import annotations

import pathlib

import pyproj

from wrf_massive.base import BBox
from wrf_massive.stages.utils import get_namelist_value

EARTH_RADIUS_M = 6370000.0  # WRF/WPS default sphere radius


def _first(v: str | list[str]) -> str:
    """Namelist fields may be a comma-separated list (one value per domain); use domain 1's value."""
    return v[0] if isinstance(v, list) else v


def compute_lambert_domain_bbox(namelist_wps_path: str | pathlib.Path, margin_deg: float = 1.0) -> BBox:
    """Compute a lat/lon bounding box covering domain 1 of a `namelist.wps` (or its Jinja2 template,
    since geogrid fields are not templated), by re-projecting its Lambert Conformal Conic geogrid
    corners back to lon/lat and padding by `margin_deg`.

    This assumes a spherical earth and ignores staggering nuances -- it's an approximation good enough
    to validate that a manually chosen forcing-data `area` isn't obviously too small, not a substitute
    for WPS/WRF's own projection math.
    """
    namelist_wps_path = pathlib.Path(namelist_wps_path)

    map_proj = _first(get_namelist_value(namelist_wps_path, "map_proj")).strip("'\"")
    if map_proj != "lambert":
        raise NotImplementedError(f"Only 'lambert' map_proj is supported, got {map_proj!r}.")

    ref_lat = float(_first(get_namelist_value(namelist_wps_path, "ref_lat")))
    ref_lon = float(_first(get_namelist_value(namelist_wps_path, "ref_lon")))
    truelat1 = float(_first(get_namelist_value(namelist_wps_path, "truelat1")))
    truelat2 = float(_first(get_namelist_value(namelist_wps_path, "truelat2")))
    stand_lon = float(_first(get_namelist_value(namelist_wps_path, "stand_lon")))
    dx = float(_first(get_namelist_value(namelist_wps_path, "dx")))
    dy = float(_first(get_namelist_value(namelist_wps_path, "dy")))
    e_we = int(_first(get_namelist_value(namelist_wps_path, "e_we")))
    e_sn = int(_first(get_namelist_value(namelist_wps_path, "e_sn")))

    proj = pyproj.Proj(proj="lcc", lat_1=truelat1, lat_2=truelat2, lat_0=ref_lat, lon_0=stand_lon, R=EARTH_RADIUS_M)
    x0, y0 = proj(ref_lon, ref_lat)

    half_width = (e_we - 1) * dx / 2
    half_height = (e_sn - 1) * dy / 2
    corners_xy = [
        (x0 - half_width, y0 - half_height),
        (x0 - half_width, y0 + half_height),
        (x0 + half_width, y0 - half_height),
        (x0 + half_width, y0 + half_height),
    ]
    corners_lonlat = [proj(x, y, inverse=True) for x, y in corners_xy]
    lons = [lon for lon, _ in corners_lonlat]
    lats = [lat for _, lat in corners_lonlat]

    return BBox(
        north=max(lats) + margin_deg,
        south=min(lats) - margin_deg,
        west=min(lons) - margin_deg,
        east=max(lons) + margin_deg,
    )


def validate_area_covers_domain(area: BBox, domain_bbox: BBox) -> None:
    """Raise `ValueError` if `area` doesn't fully contain `domain_bbox`."""
    if not (
        area.north >= domain_bbox.north
        and area.south <= domain_bbox.south
        and area.west <= domain_bbox.west
        and area.east >= domain_bbox.east
    ):
        raise ValueError(
            f"Forcing-data area {area} does not fully cover the WRF domain bounding box {domain_bbox}. "
            "Widen `Simulation.area` (or increase the margin) so the CDS request covers the whole domain."
        )
