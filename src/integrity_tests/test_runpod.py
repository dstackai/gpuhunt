import csv
from collections import Counter
from pathlib import Path

import pytest

from gpuhunt.providers.runpod import get_gpu_map


@pytest.fixture
def data_rows(catalog_dir: Path) -> list[dict]:
    file = catalog_dir / "runpod.csv"
    reader = csv.DictReader(file.open())
    return list(reader)


def select_row(rows, name: str) -> list[str]:
    return [r[name] for r in rows if r[name]]


def test_locations(data_rows):
    expected = {
        # Secure cloud
        "CA-MTL-1",
        "CA-MTL-2",
        "CA-MTL-3",
        "EU-NL-1",
        "EU-RO-1",
        "EU-SE-1",
        "EUR-IS-1",
        "EUR-IS-2",
        "US-TX-3",
        # Community cloud
        "CA",
        "CZ",
        "FR",
        "US",
    }
    locations = set(select_row(data_rows, "location"))
    # Assert most are present. Some may be missing due to low availability
    assert len(expected - locations) <= 3


def test_spot(data_rows):
    spots = select_row(data_rows, "spot")

    expected = set(("True", "False"))
    assert set(spots) == expected

    count = Counter(spots)
    for spot_key in ("True", "False"):
        assert count[spot_key] > 1


def test_gpu_present(data_rows):
    refs = set(name for _, name in get_gpu_map().values())
    gpus = set(select_row(data_rows, "gpu_name"))
    assert len(refs & gpus) > 7
