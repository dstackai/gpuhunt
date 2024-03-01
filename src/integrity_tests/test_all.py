import csv
import os
from pathlib import Path

import pytest

files = sorted(Path(os.environ["CATALOG_DIR"]).glob("*.csv"))


def catalog_name(catalog) -> str:
    return catalog.name


class TestAllCatalogs:
    @pytest.fixture(params=files, ids=catalog_name)
    def catalog(self, request):
        yield request.param

    def test_non_zero_cost(self, catalog):
        reader = csv.DictReader(catalog.open())
        for row in reader:
            assert float(row["price"]) != pytest.approx(0), str(row)
