# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helpers for the WRF stage."""

from __future__ import annotations

import pathlib
from typing import List

from wrf_massive.stages.utils import get_namelist_value


def estimate_ncpus(n_we: int, n_sn: int) -> tuple[int, int]:
    """Estimate upper and lower limits of WRF cpu counts.
    Based on https://forum.mmm.ucar.edu/threads/choosing-an-appropriate-number-of-processors.5082/
    """
    n_min = (n_we) / 100 * (n_sn) / 100
    n_max = (n_we) / 25 * (n_sn) / 25
    return max(1, int(n_min)), max(1, int(n_max))


def _domain_values(namelist_path: str | pathlib.Path, field: str, n_domains: int) -> List[int]:
    """Integer values of `field` for the first `n_domains` domains."""
    value = get_namelist_value(namelist_path, field)
    values = value if isinstance(value, list) else [value]
    return [int(v) for v in values[:n_domains]]


def estimate_ncpus_from_namelist(namelist_path: str | pathlib.Path) -> tuple[int, int]:
    """Estimate upper and lower limits of WRF cpu counts from a namelist.input file.

    Uses `e_we`/`e_sn` of the first `max_dom` domains. With nested domains, the limits are those that suit every
    domain: the lower limit of the largest and the upper limit of the smallest domain.
    """
    n_domains = int(get_namelist_value(namelist_path, "max_dom"))
    e_we = _domain_values(namelist_path, "e_we", n_domains)
    e_sn = _domain_values(namelist_path, "e_sn", n_domains)
    limits = [estimate_ncpus(we, sn) for we, sn in zip(e_we, e_sn)]
    return max(n_min for n_min, _ in limits), min(n_max for _, n_max in limits)
