import csv
from operator import itemgetter
from pathlib import Path

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> list[dict]:
    with open(catalog_dir / "lambdalabs.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("gpu", ["A10", "A100", "H100"])
def test_gpu_present(gpu: str, data_rows: list[dict]):
    assert gpu in map(itemgetter("gpu_name"), data_rows)


def test_on_demand_present(data_rows: list[dict]):
    assert "False" in map(itemgetter("spot"), data_rows)


def test_spot_not_present(data_rows: list[dict]):
    assert "True" not in map(itemgetter("spot"), data_rows)


def test_locations(data_rows: list[dict]):
    expected_locations = {
        "asia-northeast-1",
        "asia-northeast-2",
        "asia-south-1",
        "australia-east-1",
        "europe-central-1",
        "me-west-1",
        "us-east-1",
        "us-east-3",
        "us-midwest-1",
        "us-south-1",
        "us-south-2",
        "us-south-3",
        "us-west-1",
        "us-west-2",
        "us-west-3",
    }
    locations = set(map(itemgetter("location"), data_rows))
    missing = expected_locations - locations
    assert not missing
