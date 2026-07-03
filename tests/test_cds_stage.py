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

    request_path = stage.get_work_dir(simulation_with_area) / "cds_request_SFC.yaml"
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
    request = yaml.safe_load((stage.get_work_dir(simulation_with_area) / "cds_request_PRES.yaml").read_text())
    assert request["pressure_level"] == ["1000", "925"]


def test_extra_params_override_defaults(simulation_with_area):
    stage = make_stage(
        requests=[
            CdsRequestSpec(
                dataset="reanalysis-cerra-single-levels",
                variables=["2m_temperature"],
                file_suffix="SFC",
                extra_params={"product_type": ["analysis"]},
            ),
        ],
    )
    stage.setup(simulation_with_area)
    request = yaml.safe_load((stage.get_work_dir(simulation_with_area) / "cds_request_SFC.yaml").read_text())
    assert request["product_type"] == ["analysis"]
