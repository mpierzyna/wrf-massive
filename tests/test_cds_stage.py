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
    _grib_reference_key,
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


@pytest.fixture
def simulation_with_area(simple_simulation) -> Simulation:
    simple_simulation.area = BBox(north=54, west=2, south=50, east=8)
    return simple_simulation


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


def test_setup_requires_area(simple_simulation):
    stage = make_stage()
    with pytest.raises(ValueError, match="Simulation.area"):
        stage.setup(simple_simulation)


def test_setup_writes_request_yaml(simulation_with_area):
    stage = make_stage()
    stage.setup(simulation_with_area)

    jan = load_req(stage, simulation_with_area, "ERA5_SFC_202501")
    assert jan["variable"] == ["2m_temperature"]
    assert jan["area"] == [54, 2, 50, 8]
    assert "pressure_level" not in jan


def test_month_split_is_waste_free(simulation_with_area):
    """Each per-month request stays inside its own month -- no cross-product bleed across the boundary."""
    stage = make_stage()
    stage.setup(simulation_with_area)

    dec = load_req(stage, simulation_with_area, "ERA5_SFC_202412")
    jan = load_req(stage, simulation_with_area, "ERA5_SFC_202501")
    assert (dec["year"], dec["month"], dec["day"]) == (["2024"], ["12"], ["31"])
    assert (jan["year"], jan["month"], jan["day"]) == (["2025"], ["01"], ["01", "02"])


def test_is_setup_reflects_request_files(simulation_with_area):
    stage = make_stage()
    assert not stage.is_setup(simulation_with_area)
    stage.setup(simulation_with_area)
    assert stage.is_setup(simulation_with_area)


def test_run_calls_cdsapi_client_per_month_splits_and_is_done(monkeypatch, simulation_with_area):
    stage = make_stage()
    stage.setup(simulation_with_area)

    period_times = {k: v for k, v in stage._periods(simulation_with_area).items()}
    events = _install_fake_cdsapi(monkeypatch, period_times)

    assert not stage.is_done(simulation_with_area)
    stage.run(simulation_with_area)

    # All jobs are submitted before any is downloaded, so they queue in parallel at CDS.
    kinds = [kind for kind, _ in events]
    assert kinds == ["submit", "submit", "download", "download"]  # two calendar months

    work_dir = stage.get_work_dir(simulation_with_area)
    # The multi-time monthly downloads are split into one file per valid time and then removed.
    assert not list(work_dir.glob("ERA5_SFC_2024*.grb")) or all(
        re.search(r"_\d{8}_\d{4}\.grb$", str(p)) for p in work_dir.glob("ERA5_SFC_*.grb")
    )
    assert not (work_dir / "ERA5_SFC_202412.grb").exists() and not (work_dir / "ERA5_SFC_202501.grb").exists()
    all_times = [t for times in period_times.values() for t in times]
    for t in all_times:
        assert (work_dir / f"ERA5_SFC_{t.year:04d}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}.grb").exists()
    assert len(list(work_dir.glob("ERA5_SFC_*_*.grb"))) == len(all_times)
    assert stage.is_done(simulation_with_area)


def test_run_skips_already_downloaded_months(monkeypatch, simulation_with_area):
    """An interrupted pull resumes: a month whose per-time files already exist is not re-requested."""
    stage = make_stage()
    stage.setup(simulation_with_area)

    # Pretend December was already fetched and split in a previous run (per-time files present).
    work_dir = stage.get_work_dir(simulation_with_area)
    for t in stage._periods(simulation_with_area)["202412"]:
        (work_dir / f"ERA5_SFC_{t.year:04d}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}.grb").write_bytes(b"x")

    period_times = {k: v for k, v in stage._periods(simulation_with_area).items()}
    events = _install_fake_cdsapi(monkeypatch, period_times)
    stage.run(simulation_with_area)

    submits = [target for kind, target in events if kind == "submit"]
    downloads = [target for kind, target in events if kind == "download"]
    assert len(submits) == 1 and len(downloads) == 1
    assert downloads[0].endswith("ERA5_SFC_202501.grb")


def test_run_resumes_split_from_existing_monthly(monkeypatch, simulation_with_area):
    """A month downloaded but not yet split on a previous run is split without being re-requested."""
    stage = make_stage()
    stage.setup(simulation_with_area)

    period_times = {k: v for k, v in stage._periods(simulation_with_area).items()}
    work_dir = stage.get_work_dir(simulation_with_area)
    _write_multi_time_grib(work_dir / "ERA5_SFC_202412.grb", period_times["202412"], edition=1)

    events = _install_fake_cdsapi(monkeypatch, period_times)
    stage.run(simulation_with_area)

    submits = [target for kind, target in events if kind == "submit"]
    assert len(submits) == 1  # only January is re-requested; December is only split
    for t in period_times["202412"]:
        assert (work_dir / f"ERA5_SFC_{t.year:04d}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}.grb").exists()
    assert not (work_dir / "ERA5_SFC_202412.grb").exists()  # aggregate removed after split
    assert stage.is_done(simulation_with_area)


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


def test_grib_reference_key_reads_both_editions():
    t = pd.Timestamp("2019-11-05T21")
    assert _grib_reference_key(_grib1_message(t)) == ("20191105_2100", 64)
    assert _grib_reference_key(_grib2_message(t)) == ("20191105_2100", 64)


def test_pressure_level_request_includes_levels(simulation_with_area):
    stage = make_stage(
        requests=[
            CdsRequestSpec(
                dataset="reanalysis-era5-pressure-levels",
                variables=["temperature"],
                levels=["1000", "925"],
                file_suffix="PRES",
            ),
        ],
    )
    stage.setup(simulation_with_area)
    assert load_req(stage, simulation_with_area, "ERA5_PRES_202501")["pressure_level"] == ["1000", "925"]


def test_extra_params_override_defaults(simulation_with_area):
    stage = make_stage(
        requests=[
            CdsRequestSpec(
                dataset="reanalysis-cerra-single-levels",
                variables=["2m_temperature"],
                file_suffix="SFC",
                extra_params={"data_type": ["reanalysis"]},
            ),
        ],
    )
    stage.setup(simulation_with_area)
    assert load_req(stage, simulation_with_area, "ERA5_SFC_202501")["data_type"] == ["reanalysis"]


def test_product_type_field(simulation_with_area):
    stage = make_stage(
        requests=[
            CdsRequestSpec(
                dataset="reanalysis-cerra-single-levels",
                variables=["2m_temperature"],
                file_suffix="SFC",
                product_type=["analysis"],
            ),
        ],
    )
    stage.setup(simulation_with_area)
    assert load_req(stage, simulation_with_area, "ERA5_SFC_202501")["product_type"] == ["analysis"]


def test_use_area_false_omits_area(simulation_with_area):
    stage = make_stage(
        requests=[
            CdsRequestSpec(
                dataset="reanalysis-cerra-single-levels",
                variables=["2m_temperature"],
                file_suffix="SFC",
                use_area=False,
            ),
        ],
    )
    stage.setup(simulation_with_area)
    assert "area" not in load_req(stage, simulation_with_area, "ERA5_SFC_202501")


def test_use_area_false_does_not_require_area(simple_simulation):
    """A full-domain (use_area=False) request must not demand Simulation.area."""
    stage = make_stage(
        requests=[
            CdsRequestSpec(
                dataset="reanalysis-cerra-single-levels",
                variables=["2m_temperature"],
                file_suffix="SFC",
                use_area=False,
            ),
        ],
    )
    stage.setup(simple_simulation)  # simple_simulation has no area; must not raise
    assert stage.is_setup(simple_simulation)


def test_shared_work_dir_requests_do_not_collide(simulation_with_area):
    """Two stages sharing a work_dir with overlapping file suffixes must keep separate request sidecars."""
    cerra = PullCdsStage(
        work_dir="forcing",
        prefix="CERRA",
        requests=[
            CdsRequestSpec(
                dataset="reanalysis-cerra-single-levels",
                variables=["2m_temperature"],
                file_suffix="SFC",
                product_type=["analysis"],
                use_area=False,
            ),
        ],
    )
    era5 = PullCdsStage(
        work_dir="forcing",
        prefix="ERA5",
        requests=[
            CdsRequestSpec(dataset="reanalysis-era5-single-levels", variables=["skin_temperature"], file_suffix="SFC"),
        ],
    )
    cerra.setup(simulation_with_area)
    era5.setup(simulation_with_area)  # must not clobber the CERRA request written above

    cerra_req = load_req(cerra, simulation_with_area, "CERRA_SFC_202501")
    era5_req = load_req(era5, simulation_with_area, "ERA5_SFC_202501")
    assert cerra_req["product_type"] == ["analysis"] and "area" not in cerra_req
    assert era5_req["product_type"] == ["reanalysis"] and "area" in era5_req
    assert cerra.is_setup(simulation_with_area) and era5.is_setup(simulation_with_area)
