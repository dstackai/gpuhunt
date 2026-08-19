import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.hotaisle import HotAisleProvider
from integrity_tests.base import OffersIntegrityTests


class TestHotAisleOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return HotAisleProvider.from_env().get()
