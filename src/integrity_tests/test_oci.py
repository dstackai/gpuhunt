import csv
from operator import itemgetter
from pathlib import Path

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> list[dict]:
    with open(catalog_dir / "oci.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("gpu", ["P100", "V100", "A10", "A100", ""])
def test_gpu_present(gpu: str, data_rows: list[dict]):
    assert gpu in map(itemgetter("gpu_name"), data_rows)


def test_on_demand_present(data_rows: list[dict]):
    assert "False" in map(itemgetter("spot"), data_rows)


def test_spot_present(data_rows: list[dict]):
    assert "True" in map(itemgetter("spot"), data_rows)


def test_spots_contain_flag(data_rows: list[dict]):
    for row in data_rows:
        assert (row["spot"] == "True") == ("oci-spot" in row["flags"]), row


@pytest.mark.parametrize("prefix", ["VM.Standard", "BM.Standard", "VM.GPU", "BM.GPU"])
def test_family_present(prefix: str, data_rows: list[dict]):
    assert any(name.startswith(prefix) for name in map(itemgetter("instance_name"), data_rows))


def test_quantity_decreases_as_query_complexity_increases(data_rows: list[dict]):
    zero_or_one_gpu = list(filter(lambda row: int(row["gpu_count"]) in (0, 1), data_rows))
    zero_gpu = list(filter(lambda row: int(row["gpu_count"]) == 0, data_rows))
    one_gpu = list(filter(lambda row: int(row["gpu_count"]) == 1, data_rows))

    assert len(data_rows) > len(zero_or_one_gpu)
    assert len(zero_or_one_gpu) > len(zero_gpu)
    assert len(zero_gpu) > len(one_gpu)
