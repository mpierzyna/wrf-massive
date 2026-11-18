# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

from wrf_massive.stages.wps.wps import is_metgrid_successful

SUCCESS = "2026-11-18 03:11:01.787 ---  *** Successful completion of program metgrid.exe ***\n"


def test_is_metgrid_successful(tmp_path):
    """Needs met_em files and the success line in metgrid.log (serial) or rank 0's metgrid.log.0000 (MPI)."""
    for log_name in ["metgrid.log", "metgrid.log.0000"]:
        wps_dir = tmp_path / log_name
        wps_dir.mkdir()
        (wps_dir / log_name).write_text("Processing domain 1 of 1\n" + SUCCESS)
        assert not is_metgrid_successful(wps_dir)  # success logged but no met_em files
        (wps_dir / "met_em.d01.2020-07-01_00:00:00.nc").write_text("x")
        assert is_metgrid_successful(wps_dir)

    wps_dir = tmp_path / "failed"
    wps_dir.mkdir()
    (wps_dir / "met_em.d01.2020-07-01_00:00:00.nc").write_text("x")
    assert not is_metgrid_successful(wps_dir)  # met_em but no log
    (wps_dir / "metgrid.log.0000").write_text("Processing domain 1 of 1\n")  # crashed part-way
    assert not is_metgrid_successful(wps_dir)
    (wps_dir / "metgrid.log.0001").write_text(SUCCESS)  # only rank 0 is authoritative, like run_wps.sh
    assert not is_metgrid_successful(wps_dir)
