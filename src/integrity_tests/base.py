import csv
from pathlib import Path

import pytest

from gpuhunt import CatalogItem
from gpuhunt._internal.storage import CATALOG_V2_FIELDS, item_from_row

# Fields that are allowed to be empty, including empty strings or empty lists
OPTIONAL_CATALOG_ITEM_FIELDS = ["gpu_name", "gpu_memory", "gpu_vendor", "disk_size", "flags"]


class OffersIntegrityTests:
    """
    Invariants that offers must satisfy regardless of the provider.

    Subclasses provide the `offers` fixture. Online providers fetch offers from the API,
    offline providers parse them from the published catalog file, see
    `CatalogFileIntegrityTests`.
    """

    def test_offers_present(self, offers: list[CatalogItem]) -> None:
        assert offers

    def test_name_and_location_present(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            assert offer.instance_name, str(offer)
            assert offer.location, str(offer)

    def test_price_positive(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            assert offer.price > 0, str(offer)

    def test_resources_positive(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            assert offer.cpu > 0, str(offer)
            assert offer.memory > 0, str(offer)
            assert offer.disk_size is None or offer.disk_size > 0, str(offer)

    def test_gpu_consistent(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            if offer.gpu_count > 0:
                assert offer.gpu_name, str(offer)
                assert offer.gpu_memory and offer.gpu_memory > 0, str(offer)
                assert offer.gpu_vendor, str(offer)
            else:
                assert offer.gpu_count == 0, str(offer)
                assert offer.gpu_name is None, str(offer)
                assert offer.gpu_memory is None, str(offer)
                assert offer.gpu_vendor is None, str(offer)


class CatalogFileIntegrityTests(OffersIntegrityTests):
    """
    `OffersIntegrityTests` for a published catalog file, plus checks specific to the CSV
    format. Subclasses set `CATALOG_NAME` to the name of the catalog file without the
    extension, which is also the provider name.
    """

    CATALOG_NAME: str

    @pytest.fixture
    def rows(self, catalog_dir: Path) -> list[dict[str, str]]:
        with (catalog_dir / f"{self.CATALOG_NAME}.csv").open() as f:
            return list(csv.DictReader(f))

    @pytest.fixture
    def offers(self, rows: list[dict[str, str]]) -> list[CatalogItem]:
        return [item_from_row(row, provider=self.CATALOG_NAME) for row in rows]

    @pytest.mark.parametrize(
        "field",
        [f for f in CATALOG_V2_FIELDS if f not in OPTIONAL_CATALOG_ITEM_FIELDS],
    )
    def test_field_present(self, rows: list[dict[str, str]], field: str) -> None:
        for row in rows:
            assert row[field], str(row)

    # `item_from_row` backfills a missing vendor as Nvidia (Google for TPUs) so that catalogs
    # published before the `gpu_vendor` column existed still load. Newly generated catalogs
    # must not rely on that, and the column is only optional for CPU offers, so the
    # requirement is asserted on the raw row.
    def test_gpu_vendor_present(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            if int(row["gpu_count"]) > 0:
                assert row["gpu_vendor"], str(row)

    def test_spot_boolean(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            assert row["spot"] in ("True", "False"), str(row)
