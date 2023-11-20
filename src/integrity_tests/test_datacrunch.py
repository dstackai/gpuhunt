import csv
import itertools
from collections import Counter
from pathlib import Path
from typing import Any, List

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> List[dict]:
    file = catalog_dir / "datacrunch.csv"
    reader = csv.DictReader(file.open())
    return list(reader)


def select_row(rows, name: str) -> List[str]:
    return [r[name] for r in rows]


def test_locations(data_rows):
    expected = set(("FIN-01", "ICE-01"))
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
    refs = ("A100", "V100", "A6000", "RTX6000")
    gpus = select_row(data_rows, "gpu_name")
    combinations = ((ref in gpu) for ref, gpu in itertools.product(refs, gpus))
    assert len(list(filter(None, combinations))) > 0
