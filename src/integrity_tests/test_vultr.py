import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.vultr import VultrProvider
from integrity_tests.base import OffersIntegrityTests


class TestVultrOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return VultrProvider.from_env().get()

    def test_gpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count > 0 for o in offers)

    def test_cpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == 0 for o in offers)

    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)
