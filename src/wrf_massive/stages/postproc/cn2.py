from __future__ import annotations

from typing import Dict, List

import numpy as np
import xarray as xr

from wrf_massive.stages.postproc import PostProcFn, PostProcStage
from wrf_massive.stages.postproc.base import TVarList
from wrf_massive.stages.postproc.precip import fn_int_water_paths


def _get_ct2_hb15(*, var_theta, Lm):
    """Variance-based CT2 parameterization from He and Basu (2015).

    Parameters
    ----------
    var_theta : float
        Variance of potential temperature, K^2.
    Lm : float
        Master length scale, m.

    Returns
    -------
    CT2 : float
        CT2 estimate, K^2 m^(-2/3).
    """
    # Coefficents from LES
    B1 = 24
    B2 = 15

    # Clip Lm to avoid division by zero
    Lm = np.clip(Lm, a_min=1e-4, a_max=None)

    ct2 = 3.2 * B1 ** (1 / 3) / B2 * Lm ** (-2 / 3) * var_theta
    return ct2


def _gladstone_cn2_simple(*, ct2, p_hPa, t_K):
    """Simplified Gladstone equation without humidity correction."""
    cn2 = (7.9e-5 * p_hPa / t_K**2) ** 2 * ct2
    return cn2


def _get_ct2_cn2(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """Calculate CT2 and Cn2 from dataset containing TSQ, EL_PBL, p, and tk."""
    ct2 = _get_ct2_hb15(var_theta=ds["TSQ"], Lm=ds["EL_PBL"])
    ct2.name = "ct2"
    ct2.attrs = {"units": "K^2 m^(-2/3)", "long_name": "CT2, parameterized acc. to He and Basu (2015)"}

    cn2 = _gladstone_cn2_simple(ct2=ct2, p_hPa=ds["p"] / 100, t_K=ds["tk"])
    cn2.name = "cn2"
    cn2.attrs = {"units": "m^(-2/3)", "long_name": "Cn2 from CT2 and Gladstone eqn."}

    return {
        "ct2": ct2,
        "cn2": cn2,
    }


fn_ct2_cn2 = PostProcFn(
    fn=_get_ct2_cn2,
    requires=["TSQ", ("EL_PBL", "bottom_top_stag"), "p", "tk"],
    returns=["ct2", "cn2"],
)


class Cn2PostProcStage(PostProcStage):
    """Post-processing stage for typical CT2/Cn2 postprocessing."""

    extract_vars: TVarList = [
        "z",
        "HGT",
        "p",
        "u_met",
        "v_met",
        "w",
        "th",  # potential temperature
        # "tk",
        "rh",
        "PBLH",
        "LANDMASK",
        "mdbz",  # maximum reflectivity
        "slp",
        "T2",
        # "TH2",
        "U10",
        "V10",
        "LH",
        "HFX",
        "UST",
        # "ZNT",
        # "Z0",
        "QKE",
        ("EL_PBL", "bottom_top_stag"),
        "TSQ",
    ]

    postproc_fns: List[PostProcFn] = [
        fn_ct2_cn2,
        fn_int_water_paths,
    ]

    file_suffix: str = "cn2"
