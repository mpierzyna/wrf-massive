from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr

from wrf_massive.stages.postproc.base import PostProcFn


def get_ct2_hb15(*, var_theta, Lm):
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


def gladstone_cn2_simple(*, ct2, p_hPa, t_K):
    """Simplified Gladstone equation without humidity correction."""
    cn2 = (7.9e-5 * p_hPa / t_K**2) ** 2 * ct2
    return cn2


def get_ct2_cn2_fn() -> PostProcFn:
    """Return a PostProcFn that calculates CT2 and Cn2 from TSQ, EL_PBL, p, and tk."""

    def _fn(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
        """Calculate CT2 and Cn2 from dataset containing TSQ, EL_PBL, p, and tk."""
        ct2 = get_ct2_hb15(var_theta=ds["TSQ"], Lm=ds["EL_PBL"])
        ct2.name = "ct2"
        ct2.attrs = {"units": "K^2 m^(-2/3)", "long_name": "CT2, parameterized acc. to He and Basu (2015)"}

        cn2 = gladstone_cn2_simple(ct2=ct2, p_hPa=ds["p"] / 100, t_K=ds["tk"])
        cn2.name = "cn2"
        cn2.attrs = {"units": "m^(-2/3)", "long_name": "Cn2 from CT2 and Gladstone eqn."}

        return {
            "ct2": ct2,
            "cn2": cn2,
        }

    return PostProcFn(
        fn=_fn,
        requires=["TSQ", ("EL_PBL", "bottom_top_stag"), "p", "tk"],
        returns=["ct2", "cn2"],
    )
