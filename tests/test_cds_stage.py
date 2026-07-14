import pathlib
import sys
import types

import pytest
import yaml
from fixtures import simple_simulation

from wrf_massive.base import BBox, Simulation
from wrf_massive.stages.forcing.cds import CdsRequestSpec, PullCdsStage

# simple_simulation spans begin_w_warmup 2024-12-31T12 -> end 2025-01-02T00, i.e. two calendar months, so a
# pull is split into one request per month: suffix _202412 (Dec 31) and _202501 (Jan 1-2).


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


def _install_fake_cdsapi(monkeypatch) -> list:
    """Install a fake `cdsapi` matching the async flow: `Client(...).retrieve(dataset, request)` submits and
    returns a handle; `handle.download(target)` writes the file. Returns a shared list of
    `("submit", dataset)` / `("download", target)` events in call order.
    """
    events: list = []

    class FakeHandle:
        def __init__(self, dataset):
            self.request_id = f"rid-{dataset}"

        def download(self, target):
            events.append(("download", target))
            pathlib.Path(target).write_bytes(b"fake-grib-bytes")
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


def test_run_calls_cdsapi_client_per_month_and_is_done(monkeypatch, simulation_with_area):
    stage = make_stage()
    stage.setup(simulation_with_area)

    events = _install_fake_cdsapi(monkeypatch)

    assert not stage.is_done(simulation_with_area)
    stage.run(simulation_with_area)

    # All jobs are submitted before any is downloaded, so they queue in parallel at CDS.
    kinds = [kind for kind, _ in events]
    assert kinds == ["submit", "submit", "download", "download"]  # two calendar months
    downloads = sorted(target for kind, target in events if kind == "download")
    assert downloads[0].endswith("ERA5_SFC_202412.grb")
    assert downloads[1].endswith("ERA5_SFC_202501.grb")
    assert stage.is_done(simulation_with_area)


def test_run_skips_already_downloaded_months(monkeypatch, simulation_with_area):
    """An interrupted pull resumes: a month whose grib already exists is not re-requested."""
    stage = make_stage()
    stage.setup(simulation_with_area)

    # Pretend the December file was already fetched in a previous run.
    dec_target = stage.get_work_dir(simulation_with_area) / "ERA5_SFC_202412.grb"
    dec_target.write_bytes(b"already-here")

    events = _install_fake_cdsapi(monkeypatch)
    stage.run(simulation_with_area)

    submits = [target for kind, target in events if kind == "submit"]
    downloads = [target for kind, target in events if kind == "download"]
    assert len(submits) == 1 and len(downloads) == 1
    assert downloads[0].endswith("ERA5_SFC_202501.grb")


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
