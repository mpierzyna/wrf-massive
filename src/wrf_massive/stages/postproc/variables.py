# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Variable names in post-processing and the wrf-python derived variables.

Naming convention
-----------------
- ALL-CAPS names (`TSQ`, `PH`, `U10`, ...) are WRF fields. They are read from the wrfout file, so WRF must write
  them, i.e. they must be listed in `myoutfields.txt` for the post-processed stream.
- lower-case names (`p`, `tk`, `u_met`, `cn2`, ...) are derived variables: either one of the wrf-python variables in
  `DERIVED_VARS` below, read from the wrfout file like a WRF field, or returned by a post-processing function.
"""

from typing import Dict, List, NamedTuple, Tuple

# Type alias for list of variable names to extract from wrfout files.
# If tuple, first element is variable name, second element is dimension along which to destagger.
TVarList = List[str | Tuple[str, str]]

# WRF fields every post-processing run needs: wrf-python attaches them as lat/lon coordinates to everything it
# reads, raw WRF fields included, and fails without them.
REQUIRED_WRF_FIELDS = ("XLAT", "XLONG")


def var_name(v: str | Tuple[str, str]) -> str:
    """Name of a `TVarList` entry, without its optional destagger dimension."""
    return v[0] if isinstance(v, tuple) else v


def is_wrf_field(name: str) -> bool:
    """ALL-CAPS names are WRF fields, everything else is a derived variable."""
    return name.isupper()


class DerivedVar(NamedTuple):
    """A variable computed by `wrf.getvar(getvar)`, optionally one component of it (e.g. `uvmet[0]` is `u_met`)."""

    getvar: str
    requires: List[str]  # WRF fields wrf-python reads for it
    index: int | None = None  # component along the first dimension, for variables wrf-python returns stacked


# `requires` are the WRF fields wrf-python 1.4.2 reads (see the `get_*` functions in `wrf/g_*.py`, e.g.
# `wrf.g_temp.get_tk` extracts `T`, `P`, `PB`), plus the template field its metadata decorator copies coordinates from
# (`P` for heights and 3D winds, see `wrf/metadecorators.py`). Optional fields wrf-python uses only if present (e.g.
# `QSNOW`/`QGRAUP` for `mdbz`) are not listed. `tests/test_postproc_stage.py::test_derived_requires_are_sufficient`
# reads every entry from a test fixture reduced to the listed fields, so a wrong entry or a changed wrf-python fails the
# tests. Every entry must be verifiable by one of the fixtures in `tests/` (`mdbz` needs QRAIN, 2 m/10 m vars PSFC).
_THERMO = ["T", "P", "PB"]  # full potential temperature and pressure
_GEOPT = ["PH", "PHB"]  # full geopotential (staggered)
_HEIGHT = [*_GEOPT, "HGT", "P"]  # heights: geopotential + terrain, coordinates copied from P
_UVMET = ["U", "V", "XLAT", "XLONG", "P"]  # XLAT/XLONG rotate grid-relative U/V to earth-relative
_UVMET10 = ["U10", "V10", "XLAT", "XLONG", "PSFC"]  # as above, coordinates copied from PSFC

DERIVED_VARS: Dict[str, DerivedVar] = {
    "z": DerivedVar("z", _HEIGHT),  # height (MSL) of mass levels, m
    "height_agl": DerivedVar("height_agl", _HEIGHT),  # height above ground, m
    "p": DerivedVar("p", ["P", "PB"]),  # pressure, Pa
    "pressure": DerivedVar("pressure", ["P", "PB"]),  # pressure, hPa
    "th": DerivedVar("th", ["T"]),  # potential temperature, K
    "tk": DerivedVar("tk", _THERMO),  # temperature, K
    "tc": DerivedVar("tc", _THERMO),  # temperature, degC
    "tv": DerivedVar("tv", [*_THERMO, "QVAPOR"]),  # virtual temperature, K
    "rh": DerivedVar("rh", [*_THERMO, "QVAPOR"]),  # relative humidity, %
    "td": DerivedVar("td", ["P", "PB", "QVAPOR"]),  # dew point temperature, degC
    "rh2": DerivedVar("rh2", ["T2", "PSFC", "Q2"]),  # 2 m relative humidity, %
    "td2": DerivedVar("td2", ["PSFC", "Q2"]),  # 2 m dew point temperature, degC
    "slp": DerivedVar("slp", [*_THERMO, "QVAPOR", *_GEOPT]),  # sea level pressure, hPa
    "pw": DerivedVar("pw", [*_THERMO, "QVAPOR", *_GEOPT]),  # precipitable water, kg m-2
    "mdbz": DerivedVar("mdbz", [*_THERMO, "QVAPOR", "QRAIN"]),  # maximum reflectivity, dBZ
    "w": DerivedVar("wa", ["W", "P"]),  # vertical wind on mass levels, m s-1
    # earth-relative winds (rotation matters for Lambert conformal and polar stereographic grids)
    "u_met": DerivedVar("uvmet", _UVMET, index=0),
    "v_met": DerivedVar("uvmet", _UVMET, index=1),
    "u10_met": DerivedVar("uvmet10", _UVMET10, index=0),
    "v10_met": DerivedVar("uvmet10", _UVMET10, index=1),
}


def wrf_fields_of(name: str) -> List[str]:
    """WRF fields needed to read variable `name` from a wrfout file: itself, or what wrf-python needs to derive it."""
    if is_wrf_field(name):
        return [name]
    if name not in DERIVED_VARS:
        raise ValueError(
            f"Unknown derived variable '{name}': it is neither returned by a postproc fn of the stage nor a wrf-python "
            f"derived variable ({', '.join(DERIVED_VARS)}). ALL-CAPS names are WRF fields, lower-case names derived "
            f"variables."
        )
    return DERIVED_VARS[name].requires
