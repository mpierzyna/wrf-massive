# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from wrf_massive.stages.wps.wps import is_metgrid_successful

SUCCESS = "2026-11-18 03:11:01.787 ---  *** Successful completion of program metgrid.exe ***\n"


@pytest.mark.parametrize("log_name", ["metgrid.log", "metgrid.log.0000"])
def test_is_metgrid_successful_serial_and_mpi_log(tmp_path, log_name):
    (tmp_path / log_name).write_text("Processing domain 1 of 1\n" + SUCCESS)
    assert not is_metgrid_successful(tmp_path)  # success logged but no met_em files

    (tmp_path / "met_em.d01.2020-07-01_00:00:00.nc").write_text("x")
    assert is_metgrid_successful(tmp_path)


def test_is_metgrid_successful_requires_success_marker(tmp_path):
    (tmp_path / "met_em.d01.2020-07-01_00:00:00.nc").write_text("x")
    assert not is_metgrid_successful(tmp_path)  # met_em but no log

    (tmp_path / "metgrid.log.0000").write_text("Processing domain 1 of 1\n")  # crashed part-way
    assert not is_metgrid_successful(tmp_path)

    (tmp_path / "metgrid.log.0001").write_text(SUCCESS)  # only rank 0 is authoritative, like run_wps.sh
    assert not is_metgrid_successful(tmp_path)
