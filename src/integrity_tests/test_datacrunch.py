import csv
from collections import Counter
from pathlib import Path
from typing import List

import pytest

from gpuhunt.providers.datacrunch import ALL_AMD_GPUS, GPU_MAP


@pytest.fixture
def data_rows(catalog_dir: Path) -> List[dict]:
    file = catalog_dir / "datacrunch.csv"
    reader = csv.DictReader(file.open())
    return list(reader)


def select_row(rows, name: str) -> List[str]:
    return [r[name] for r in rows if r[name]]


def test_locations(data_rows):
    expected = {
        "FIN-01",
        "FIN-02",
        "ICE-01",
    }
    locations = select_row(data_rows, "location")
    assert set(locations) == expected

    count = Counter(locations)
    for loc in expected:
        assert count[loc] > 1


def test_spot(data_rows):
    spots = select_row(data_rows, "spot")

    expected = set(("True", "False"))
    assert set(spots) == expected

    count = Counter(spots)
    for spot_key in ("True", "False"):
        assert count[spot_key] > 1


def test_price(data_rows):
    prices = select_row(data_rows, "price")
    assert min(float(p) for p in prices) > 0


def test_gpu_present(data_rows):
    refs = [name for name in GPU_MAP.values() if name not in ALL_AMD_GPUS]
    gpus = select_row(data_rows, "gpu_name")
    assert set(gpus) == set(refs)
