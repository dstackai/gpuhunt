import csv
from collections import Counter
from pathlib import Path
from typing import List

import pytest

from gpuhunt.providers.cudo import GPU_MAP


@pytest.fixture
def data_rows(catalog_dir: Path) -> List[dict]:
    print(catalog_dir)
    file = catalog_dir / "cudo.csv"
    reader = csv.DictReader(file.open())
    return list(reader)


def select_row(rows, name: str) -> List[str]:
    return [r[name] for r in rows]


@pytest.mark.xfail
def test_locations(data_rows):
    expected = {
        "no-luster-1",
        "se-smedjebacken-1",
        "se-stockholm-1",
        "us-newyork-1",
        "us-santaclara-1",
    }
    locations = select_row(data_rows, "location")
    assert set(locations) == expected

    count = Counter(locations)
    for loc in expected:
        assert count[loc] > 1


def test_price(data_rows):
    prices = select_row(data_rows, "price")
    assert min(float(p) for p in prices) > 0


def test_gpu_present(data_rows):
    refs = GPU_MAP.values()
    gpus = select_row(data_rows, "gpu_name")
    assert all(i in refs for i in gpus)
