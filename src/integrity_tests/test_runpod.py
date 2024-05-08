import csv
from collections import Counter
from pathlib import Path
from typing import List

import pytest

from gpuhunt.providers.runpod import GPU_MAP


@pytest.fixture
def data_rows(catalog_dir: Path) -> List[dict]:
    file = catalog_dir / "runpod.csv"
    reader = csv.DictReader(file.open())
    return list(reader)


def select_row(rows, name: str) -> List[str]:
    return [r[name] for r in rows if r[name]]


def test_locations(data_rows):
    expected = {
        "CA-MTL-1",
        "CA-MTL-2",
        "CA-MTL-3",
        "EU-NL-1",
        "EU-RO-1",
        "EU-SE-1",
        "EUR-IS-1",
        "EUR-IS-2",
        "US-GA-1",
        "US-OR-1",
        "US-TX-3",
    }
    locations = set(select_row(data_rows, "location"))
    assert len(locations) >= len(expected) - 3


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
    refs = set(name for name in GPU_MAP.values())
    gpus = set(select_row(data_rows, "gpu_name"))
    assert len(refs & gpus) > 7
