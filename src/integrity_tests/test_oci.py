import pytest

from gpuhunt import CatalogItem
from integrity_tests.base import CatalogFileIntegrityTests


class TestOCICatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "oci"

    @pytest.mark.parametrize("gpu", ["P100", "V100", "A10", "A100"])
    def test_gpu_present(self, gpu: str, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_name == gpu for o in offers)

    def test_cpu_offer_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == 0 for o in offers)

    def test_on_demand_present(self, offers: list[CatalogItem]) -> None:
        assert any(not o.spot for o in offers)

    def test_spot_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.spot for o in offers)

    def test_spots_contain_flag(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            assert offer.spot == ("oci-spot" in offer.flags), str(offer)

    @pytest.mark.parametrize("prefix", ["VM.Standard", "BM.Standard", "VM.GPU", "BM.GPU"])
    def test_family_present(self, prefix: str, offers: list[CatalogItem]) -> None:
        assert any(o.instance_name.startswith(prefix) for o in offers)

    def test_quantity_decreases_as_query_complexity_increases(
        self, offers: list[CatalogItem]
    ) -> None:
        zero_or_one_gpu = [o for o in offers if o.gpu_count in (0, 1)]
        zero_gpu = [o for o in offers if o.gpu_count == 0]
        one_gpu = [o for o in offers if o.gpu_count == 1]

        assert len(offers) > len(zero_or_one_gpu)
        assert len(zero_or_one_gpu) > len(zero_gpu)
        assert len(zero_gpu) > len(one_gpu)
