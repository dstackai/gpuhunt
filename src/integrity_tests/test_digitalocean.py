import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.digitalocean import DigitalOceanProvider
from integrity_tests.base import OffersIntegrityTests


class TestDigitalOceanOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return DigitalOceanProvider.from_env().get()

    # Basic shared droplets, available to every account
    def test_shared_droplets_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.instance_name.startswith("s-") for o in offers)

    def test_cpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == 0 for o in offers)

    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)
