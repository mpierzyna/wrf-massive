from typing import Dict

import xarray as xr

from wrf_massive.stages.postproc import PostProcFn


def _get_q_path_vars(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """Compute column-integrated water paths (kg/m^2) for rain, snow, and vapor."""
    # Compute density of moist air
    Rd = 287.05
    g = 9.81
    rho = ds["p"] / (Rd * ds["tv"])  # kg/m^3

    # Compute dz between levels
    z_w = (ds["PH"] + ds["PHB"]) / g  # geopotential height at w-levels (staggered)
    dz = z_w.diff("bottom_top_stag").rename({"bottom_top_stag": "bottom_top"})

    def column_path(q: xr.DataArray, long_name: str) -> xr.DataArray:
        """Compute column path (kg/m^2) for a given mixing ratio q (kg/kg)."""
        res = (q * rho * dz).sum(dim="bottom_top")
        res.attrs = {"units": "kg/m^2", "long_name": long_name}
        return res

    return {
        "rwp": column_path(ds["QRAIN"], "Rain Water Path"),
        "swp": column_path(ds["QSNOW"], "Snow Water Path"),
        "iwv": column_path(ds["QVAPOR"], "Integrated Water Vapor"),
    }


fn_int_water_paths = PostProcFn(
    fn=_get_q_path_vars,
    requires=["QRAIN", "QSNOW", "QVAPOR", "p", "tv", "PH", "PHB"],
    returns=["rwp", "swp", "iwv"],
)
