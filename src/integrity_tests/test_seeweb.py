import pytest

from gpuhunt import AcceleratorVendor, CatalogItem
from gpuhunt.providers.seeweb import SeewebProvider
from integrity_tests.base import OffersIntegrityTests


class TestSeewebOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return SeewebProvider.from_env().get()

    # CPU-only plans are outside the scope of the NVIDIA GPU provider
    def test_all_offers_have_gpus(self, offers: list[CatalogItem]) -> None:
        assert all(o.gpu_count > 0 for o in offers)

    def test_gpu_vendor_nvidia(self, offers: list[CatalogItem]) -> None:
        vendors = {o.gpu_vendor for o in offers}
        assert vendors == {AcceleratorVendor.NVIDIA}

    def test_disk_size_present(self, offers: list[CatalogItem]) -> None:
        assert all(o.disk_size for o in offers)

    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)
