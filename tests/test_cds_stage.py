# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import re
import sys
import types

import pandas as pd
import pytest
import yaml
from fixtures import simple_simulation

from wrf_massive.base import BBox, Simulation
from wrf_massive.stages.forcing.cds import (
    CdsRequestSpec,
    PullCdsStage,
    split_grib_by_reference_time,
)

# simple_simulation spans begin_w_warmup 2024-12-31T12 -> end 2025-01-02T00, i.e. two calendar months, so a
# pull is split into one request per month: suffix _202412 (Dec 31) and _202501 (Jan 1-2).


def _grib2_message(t: pd.Timestamp, size: int = 64) -> bytes:
    """A minimal GRIB2 message carrying reference time `t` at the identification-section offsets."""
    m = bytearray(b"\x00" * size)
    m[0:4] = b"GRIB"
    m[7] = 2  # edition
    m[8:16] = size.to_bytes(8, "big")  # total length
    s1 = 16
    m[s1 + 12 : s1 + 14] = int(t.year).to_bytes(2, "big")
    m[s1 + 14], m[s1 + 15], m[s1 + 16], m[s1 + 17] = t.month, t.day, t.hour, t.minute
    return bytes(m)


def _grib1_message(t: pd.Timestamp, size: int = 64) -> bytes:
    """A minimal GRIB1 message carrying reference time `t` at the PDS offsets (year-of-century + century)."""
    m = bytearray(b"\x00" * size)
    m[0:4] = b"GRIB"
    m[4:7] = size.to_bytes(3, "big")  # total length
    m[7] = 1  # edition
    pds = 8
    m[pds + 12] = t.year % 100  # year of century
    m[pds + 13], m[pds + 14], m[pds + 15], m[pds + 16] = t.month, t.day, t.hour, t.minute
    m[pds + 24] = t.year // 100 + 1  # century
    return bytes(m)


def _write_multi_time_grib(path, times, edition: int = 1):
    build = _grib1_message if edition == 1 else _grib2_message
    pathlib.Path(path).write_bytes(b"".join(build(t) for t in times))


def _sim_with_area(sim_dir: pathlib.Path) -> Simulation:
    """Like `simple_simulation`, with an area; a separate one per case."""
    sim_dir.mkdir(parents=True)
    s = Simulation(sim_dir=sim_dir, settings={}, warmup_h=12, begin="2025-01-01", end="2025-01-02")
    s.area = BBox(north=54, west=2, south=50, east=8)
    return s


def make_stage(**overrides) -> PullCdsStage:
    defaults = dict(
        work_dir="forcing",
        prefix="ERA5",
        requests=[
            CdsRequestSpec(dataset="reanalysis-era5-single-levels", variables=["2m_temperature"], file_suffix="SFC"),
        ],
    )
    defaults.update(overrides)
    return PullCdsStage(**defaults)


def load_req(stage: PullCdsStage, sim: Simulation, name: str) -> dict:
    """Load a request sidecar by its `<prefix>_<suffix>_<YYYYMM>` name (without the `cds_request_` prefix)."""
    return yaml.safe_load((stage.get_work_dir(sim) / f"cds_request_{name}.yaml").read_text())


def _install_fake_cdsapi(monkeypatch, period_times: dict | None = None) -> list:
    """Install a fake `cdsapi` matching the async flow: `Client(...).retrieve(dataset, request)` submits and
    returns a handle; `handle.download(target)` writes the file. Returns a shared list of
    `("submit", dataset)` / `("download", target)` events in call order.

    `period_times` maps a `YYYYMM` string to the timestamps that month's download should contain, so the written
    file is a real multi-time GRIB the stage can split. Defaults to a single synthetic timestamp per month.
    """
    period_times = period_times or {}
    events: list = []

    class FakeHandle:
        def __init__(self, dataset):
            self.request_id = f"rid-{dataset}"

        def download(self, target):
            events.append(("download", target))
            period = re.search(r"_(\d{6})\.grb$", str(target)).group(1)
            times = period_times.get(period, [pd.Timestamp(f"{period[:4]}-{period[4:]}-01T00")])
            _write_multi_time_grib(target, times, edition=1)
            return target

    class FakeClient:
        def __init__(self, **kwargs):  # must accept wait_until_complete=False
            self.kwargs = kwargs

        def retrieve(self, dataset, request, target=None):
            assert target is None, "async flow submits without a target"
            events.append(("submit", dataset))
            return FakeHandle(dataset)

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))
    return events


def test_setup_writes_request_yaml(simple_simulation, tmp_path):
    """Each request spec becomes a request sidecar per month, carrying its own dataset-specific fields. Requests with
    `use_area` (the default) need `Simulation.area`, full-domain ones do not. Each per-month request stays inside its
    month, and stages sharing a work_dir with overlapping file suffixes keep separate sidecars."""
    with pytest.raises(ValueError, match="Simulation.area"):
        make_stage().setup(simple_simulation)
    full_domain = CdsRequestSpec(
        dataset="reanalysis-cerra-single-levels",
        variables=["2m_temperature"],
        file_suffix="FULL",
        product_type=["analysis"],
        use_area=False,
        extra_params={"data_type": ["reanalysis"]},
    )
    stage = make_stage(requests=[full_domain])
    stage.setup(simple_simulation)  # simple_simulation has no area; must not raise
    assert stage.is_setup(simple_simulation)

    s = _sim_with_area(tmp_path / "sim")
    stage = make_stage(
        requests=[
            CdsRequestSpec(dataset="reanalysis-era5-single-levels", variables=["2m_temperature"], file_suffix="SFC"),
            CdsRequestSpec(
                dataset="reanalysis-era5-pressure-levels",
                variables=["temperature"],
                levels=["1000", "925"],
                file_suffix="PRES",
            ),
            full_domain,
        ],
    )
    assert not stage.is_setup(s)
    stage.setup(s)
    assert stage.is_setup(s)

    sfc = load_req(stage, s, "ERA5_SFC_202501")
    assert sfc["variable"] == ["2m_temperature"]
    assert sfc["area"] == [54, 2, 50, 8]
    assert sfc["product_type"] == ["reanalysis"]
    assert "pressure_level" not in sfc
    assert load_req(stage, s, "ERA5_PRES_202501")["pressure_level"] == ["1000", "925"]
    full = load_req(stage, s, "ERA5_FULL_202501")
    assert "area" not in full
    assert full["product_type"] == ["analysis"]
    assert full["data_type"] == ["reanalysis"]

    dec, jan = load_req(stage, s, "ERA5_SFC_202412"), sfc  # no cross-product bleed across the month boundary
    assert (dec["year"], dec["month"], dec["day"]) == (["2024"], ["12"], ["31"])
    assert (jan["year"], jan["month"], jan["day"]) == (["2025"], ["01"], ["01", "02"])

    cerra = make_stage(
        work_dir="shared",
        prefix="CERRA",
        requests=[full_domain.model_copy(update={"file_suffix": "SFC", "extra_params": {}})],
    )
    era5 = make_stage(work_dir="shared")
    cerra.setup(s)
    era5.setup(s)  # must not clobber the CERRA request written above
    cerra_req, era5_req = load_req(cerra, s, "CERRA_SFC_202501"), load_req(era5, s, "ERA5_SFC_202501")
    assert cerra_req["product_type"] == ["analysis"] and "area" not in cerra_req
    assert era5_req["product_type"] == ["reanalysis"] and "area" in era5_req
    assert cerra.is_setup(s) and era5.is_setup(s)


def _per_time_file(work_dir: pathlib.Path, t: pd.Timestamp) -> pathlib.Path:
    return work_dir / f"ERA5_SFC_{t.year:04d}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}.grb"


def test_run(monkeypatch, tmp_path):
    """All months are submitted before any is downloaded (so they queue in parallel at CDS); the multi-time monthly
    downloads are split into one file per valid time and removed. An interrupted pull resumes: a month whose per-time
    files exist is not re-requested, and a month downloaded but not yet split is only split."""
    cases = ["fresh", "already_split", "not_yet_split"]
    for case in cases:
        s = _sim_with_area(tmp_path / case)
        stage = make_stage()
        stage.setup(s)
        work_dir = stage.get_work_dir(s)
        period_times = dict(stage._periods(s))
        if case == "already_split":  # December fetched and split in a previous run
            for t in period_times["202412"]:
                _per_time_file(work_dir, t).write_bytes(b"x")
        elif case == "not_yet_split":  # December downloaded in a previous run
            _write_multi_time_grib(work_dir / "ERA5_SFC_202412.grb", period_times["202412"], edition=1)

        events = _install_fake_cdsapi(monkeypatch, period_times)
        assert not stage.is_done(s)
        stage.run(s)

        kinds = [kind for kind, _ in events]
        if case == "fresh":
            assert kinds == ["submit", "submit", "download", "download"]  # two calendar months
        else:
            assert kinds == ["submit", "download"], case  # only January is requested
            assert events[-1][1].endswith("ERA5_SFC_202501.grb")
        assert not (work_dir / "ERA5_SFC_202412.grb").exists() and not (work_dir / "ERA5_SFC_202501.grb").exists()
        all_times = [t for times in period_times.values() for t in times]
        assert all(_per_time_file(work_dir, t).exists() for t in all_times), case
        assert len(list(work_dir.glob("ERA5_SFC_*_*.grb"))) == len(all_times)
        assert stage.is_done(s)


def test_split_grib_by_reference_time_roundtrip(tmp_path):
    """Splitting groups messages by reference time, preserves bytes, and handles both GRIB editions."""
    times = [pd.Timestamp("2020-07-01T00"), pd.Timestamp("2020-07-01T00"), pd.Timestamp("2020-07-01T03")]
    for edition, build in ((1, _grib1_message), (2, _grib2_message)):
        src = tmp_path / f"multi_ed{edition}.grb"
        messages = [build(t) for t in times]
        src.write_bytes(b"".join(messages))
        written = split_grib_by_reference_time(src, lambda key: tmp_path / f"ed{edition}_{key}.grb")
        assert set(written) == {"20200701_0000", "20200701_0300"}
        # Two messages share 00:00 -> concatenated in that file; one message at 03:00.
        assert written["20200701_0000"].read_bytes() == messages[0] + messages[1]
        assert written["20200701_0300"].read_bytes() == messages[2]
