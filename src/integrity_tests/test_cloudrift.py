import csv
from operator import itemgetter
from pathlib import Path

import pytest

from gpuhunt.providers.cloudrift import GPU_MAP


@pytest.fixture
def data_rows(catalog_dir: Path) -> list[dict]:
    with open(catalog_dir / "cloudrift.csv") as f:
        return list(csv.DictReader(f))


def select_row(rows, name: str) -> list[str]:
    return [r[name] for r in rows if r[name]]


def test_gpu_present(data_rows: list[dict]):
    expected_gpus = [gpu for _, gpu in GPU_MAP]
    gpus = select_row(data_rows, "gpu_name")
    gpus = list(dict.fromkeys(gpus))
    assert set(gpus).issubset(
        set(expected_gpus)
    ), f"Found unexpected GPUs: {set(gpus) - set(expected_gpus)}"


# TODO: Add 3, 4, 5, ... 8
@pytest.mark.parametrize("gpu_count", [1, 2])
def test_gpu_count_present(gpu_count: int, data_rows: list[dict]):
    assert str(gpu_count) in map(itemgetter("gpu_count"), data_rows)
