import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.crusoe import CrusoeProvider
from integrity_tests.base import OffersIntegrityTests


class TestCrusoeOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return CrusoeProvider.from_env().get()

    def test_gpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count > 0 for o in offers)

    # TODO: Publish spot offers once spot billing can be requested via the VM create API
    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)
