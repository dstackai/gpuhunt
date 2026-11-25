import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.massedcompute import MassedComputeProvider
from integrity_tests.base import OffersIntegrityTests


class TestMassedComputeOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return MassedComputeProvider.from_env().get()
