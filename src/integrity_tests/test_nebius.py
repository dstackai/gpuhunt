import csv
from operator import itemgetter
from pathlib import Path

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> list[dict]:
    with open(catalog_dir / "nebius.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("gpu", ["L40S", "H100", "H200", ""])
def test_gpu_present(gpu: str, data_rows: list[dict]):
    assert gpu in map(itemgetter("gpu_name"), data_rows)


def test_on_demand_present(data_rows: list[dict]):
    assert "False" in map(itemgetter("spot"), data_rows)


def test_spots_presented(data_rows: list[dict]):
    spot_rows = [row for row in data_rows if row["spot"] == "True"]
    assert len(spot_rows) > 0


@pytest.mark.parametrize("location", ["eu-north1", "eu-west1"])
def test_location_present(location: str, data_rows: list[dict]):
    assert location in map(itemgetter("location"), data_rows)


def test_non_zero_price(data_rows: list[dict]):
    assert all(float(p) > 0 for p in map(itemgetter("price"), data_rows))
