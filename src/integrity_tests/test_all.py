import csv
import os
from dataclasses import fields
from pathlib import Path

import pytest

from gpuhunt._internal.models import RawCatalogItem

# Fields that are allowed to be empty, including empty strings or empty lists
OPTIONAL_CATALOG_ITEM_FIELDS = ["gpu_name", "gpu_memory", "gpu_vendor", "disk_size", "flags"]

files = sorted(Path(os.environ["CATALOG_DIR"]).glob("*.csv"))


def catalog_name(catalog) -> str:
    return catalog.name


class TestAllCatalogs:
    @pytest.fixture(params=files, ids=catalog_name)
    def catalog(self, request):
        yield csv.DictReader(request.param.open())

    @pytest.mark.parametrize(
        "field",
        [f.name for f in fields(RawCatalogItem) if f.name not in OPTIONAL_CATALOG_ITEM_FIELDS],
    )
    def test_field_present(self, catalog: csv.DictReader, field: str) -> None:
        for row in catalog:
            assert row[field], str(row)

    @pytest.mark.parametrize("field", ["spot"])
    def test_boolean_field(self, catalog: csv.DictReader, field: str) -> None:
        for row in catalog:
            assert row[field] in ("True", "False"), str(row)

    def test_gpu_consistent(self, catalog: csv.DictReader) -> None:
        for row in catalog:
            if int(row["gpu_count"]) > 0:
                assert row["gpu_name"] and row["gpu_memory"] and row["gpu_vendor"], str(row)
            else:
                assert not (row["gpu_name"] or row["gpu_memory"] or row["gpu_vendor"]), str(row)

    def test_price_positive(self, catalog: csv.DictReader) -> None:
        for row in catalog:
            if "gcp-g4-preview" in row["flags"]:
                continue
            assert float(row["price"]) > 0, str(row)
