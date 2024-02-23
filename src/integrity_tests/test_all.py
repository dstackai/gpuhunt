import csv
from pathlib import Path
from typing import List

import pytest


@pytest.fixture
def catalog_files(catalog_dir: Path) -> List[Path]:
    return list(catalog_dir.glob("*.csv"))


class TestAllCatalogs:
    def test_non_zero_cost(self, catalog_files: List[Path]):
        for file in catalog_files:
            with open(file, "r") as f:
                reader = csv.DictReader(f)
                prices = [float(row["price"]) for row in reader]
            assert 0 not in prices
