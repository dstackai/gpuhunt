import csv
from operator import itemgetter
from pathlib import Path
from typing import List

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> List[dict]:
    with open(catalog_dir / "oci.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("gpu", ["P100", "V100", "A10", ""])
def test_gpu_present(gpu: str, data_rows: List[dict]):
    assert gpu in map(itemgetter("gpu_name"), data_rows)


def test_on_demand_present(data_rows: List[dict]):
    assert "False" in map(itemgetter("spot"), data_rows)


def test_vm_present(data_rows: List[dict]):
    assert any(name.startswith("VM") for name in map(itemgetter("instance_name"), data_rows))
