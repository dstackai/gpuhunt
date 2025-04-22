import csv
from operator import itemgetter
from pathlib import Path

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> list[dict]:
    with open(catalog_dir / "cloudrift.csv") as f:
        return list(csv.DictReader(f))


# TODO: Add RTX5090 and RTX6000PRO and others after evaluation
@pytest.mark.parametrize("gpu", ["RTX4090"])
def test_gpu_present(gpu: str, data_rows: list[dict]):
    assert gpu in map(itemgetter("gpu_name"), data_rows)


# TODO: Add 3, 4, 5, ... 8
@pytest.mark.parametrize("gpu_count", [1, 2])
def test_gpu_count_present(gpu_count: int, data_rows: list[dict]):
    assert str(gpu_count) in map(itemgetter("gpu_count"), data_rows)


@pytest.mark.parametrize("location", ["us-east-nc-nr-1"])
def test_location_is_present(location: str, data_rows: list[dict]):
    assert location in map(itemgetter("location"), data_rows)


def test_non_zero_price(data_rows: list[dict]):
    assert all(float(p) > 0 for p in map(itemgetter("price"), data_rows))
