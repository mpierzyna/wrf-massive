# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib

import pytest

import wrf_massive
from wrf_massive.base import BBox
from wrf_massive.stages.forcing.geo import compute_lambert_domain_bbox, validate_area_covers_domain

NAMELIST_WPS_PATH = pathlib.Path(wrf_massive.__file__).parent / "templates" / "workspace" / "namelist.tmpl.wps"


def test_lambert_domain_bbox(tmp_path):
    """The bbox of the template domain (centred on the Netherlands, ref_lat=52.109, ref_lon=5.499) is a few degrees
    wide, widened by a margin, and checked against the forcing area; non-Lambert projections are rejected."""
    bbox = compute_lambert_domain_bbox(NAMELIST_WPS_PATH, margin_deg=0.0)

    # 144x192 grid points at 2km spacing => ~288km x ~384km domain
    assert 48 < bbox.south < bbox.north < 56
    assert 0 < bbox.west < bbox.east < 11
    assert bbox.north - bbox.south == pytest.approx(4.15, abs=0.5)
    assert bbox.east - bbox.west == pytest.approx(5.98, abs=0.5)

    bbox_margin = compute_lambert_domain_bbox(NAMELIST_WPS_PATH, margin_deg=1.0)
    assert bbox_margin.north > bbox.north and bbox_margin.south < bbox.south
    assert bbox_margin.west < bbox.west and bbox_margin.east > bbox.east

    validate_area_covers_domain(BBox(north=60, west=-5, south=45, east=15), bbox_margin)  # generous: no error
    with pytest.raises(ValueError, match="does not fully cover"):
        validate_area_covers_domain(BBox(north=53, west=5, south=52, east=6), bbox_margin)

    namelist = tmp_path / "namelist.wps"
    namelist.write_text("&geogrid\n map_proj = 'mercator'\n/\n")
    with pytest.raises(NotImplementedError):
        compute_lambert_domain_bbox(namelist)
