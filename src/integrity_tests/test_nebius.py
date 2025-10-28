import csv
import json
from operator import itemgetter
from pathlib import Path

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> list[dict]:
    with open(catalog_dir / "nebius.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("gpu", ["L40S", "H100", "H200", "B200", ""])
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


def test_fabrics_unique(data_rows: list[dict]) -> None:
    for row in data_rows:
        fabrics = json.loads(row["provider_data"])["fabrics"]
        assert len(fabrics) == len(set(fabrics)), f"Duplicate fabrics in row: {row}"


def test_fabrics_on_sample_offer(data_rows: list[dict]) -> None:
    for row in data_rows:
        if (
            row["instance_name"] == "gpu-h100-sxm 8gpu-128vcpu-1600gb"
            and row["location"] == "eu-north1"
        ):
            break
    else:
        raise ValueError("Offer not found")
    fabrics = set(json.loads(row["provider_data"])["fabrics"])
    expected_fabrics = {
        "fabric-2",
        "fabric-3",
        "fabric-4",
        "fabric-6",
    }
    missing_fabrics = expected_fabrics - fabrics
    assert not missing_fabrics


def test_no_fabrics_on_sample_non_clustered_offer(data_rows: list[dict]) -> None:
    for row in data_rows:
        if (
            row["instance_name"] == "gpu-h100-sxm 1gpu-16vcpu-200gb"
            and row["location"] == "eu-north1"
        ):
            break
    else:
        raise ValueError("Offer not found")
    assert json.loads(row["provider_data"])["fabrics"] == []
