import pathlib
import sys
import types

import pytest
import yaml
from fixtures import simple_simulation

from wrf_massive.base import BBox, Simulation
from wrf_massive.stages.forcing.cds import CdsRequestSpec, PullCdsStage


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


def test_setup_requires_area(simple_simulation):
    stage = make_stage()
    with pytest.raises(ValueError, match="Simulation.area"):
        stage.setup(simple_simulation)


def test_setup_writes_request_yaml(simulation_with_area):
    stage = make_stage()
    stage.setup(simulation_with_area)

    request_path = stage.get_work_dir(simulation_with_area) / "cds_request_ERA5_SFC.yaml"
    assert request_path.exists()
    request = yaml.safe_load(request_path.read_text())
    assert request["variable"] == ["2m_temperature"]
    assert request["area"] == [54, 2, 50, 8]
    assert "pressure_level" not in request


def test_is_setup_reflects_request_files(simulation_with_area):
    stage = make_stage()
    assert not stage.is_setup(simulation_with_area)
    stage.setup(simulation_with_area)
    assert stage.is_setup(simulation_with_area)


def test_run_calls_cdsapi_client_and_is_done(monkeypatch, simulation_with_area):
    stage = make_stage()
    stage.setup(simulation_with_area)

    retrieved = []

    class FakeClient:
        def retrieve(self, dataset, request, target):
            retrieved.append((dataset, request, target))
            pathlib.Path(target).write_bytes(b"fake-grib-bytes")

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))

    assert not stage.is_done(simulation_with_area)
    stage.run(simulation_with_area)

    assert len(retrieved) == 1
    dataset, request, target = retrieved[0]
    assert dataset == "reanalysis-era5-single-levels"
    assert target.endswith("ERA5_SFC.grb")
    assert stage.is_done(simulation_with_area)


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
    request = yaml.safe_load((stage.get_work_dir(simulation_with_area) / "cds_request_ERA5_PRES.yaml").read_text())
    assert request["pressure_level"] == ["1000", "925"]


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
    request = yaml.safe_load((stage.get_work_dir(simulation_with_area) / "cds_request_ERA5_SFC.yaml").read_text())
    assert request["data_type"] == ["reanalysis"]


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
    request = yaml.safe_load((stage.get_work_dir(simulation_with_area) / "cds_request_ERA5_SFC.yaml").read_text())
    assert request["product_type"] == ["analysis"]


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
    request = yaml.safe_load((stage.get_work_dir(simulation_with_area) / "cds_request_ERA5_SFC.yaml").read_text())
    assert "area" not in request


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

    wd = cerra.get_work_dir(simulation_with_area)
    cerra_req = yaml.safe_load((wd / "cds_request_CERRA_SFC.yaml").read_text())
    era5_req = yaml.safe_load((wd / "cds_request_ERA5_SFC.yaml").read_text())
    assert cerra_req["product_type"] == ["analysis"] and "area" not in cerra_req
    assert era5_req["product_type"] == ["reanalysis"] and "area" in era5_req
    assert cerra.is_setup(simulation_with_area) and era5.is_setup(simulation_with_area)
