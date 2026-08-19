import pytest

from gpuhunt import AcceleratorVendor, CatalogItem
from gpuhunt.providers.jarvislabs import JARVISLABS_REGION_URLS, JarvisLabsProvider
from integrity_tests.base import OffersIntegrityTests


class TestJarvisLabsOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return JarvisLabsProvider.from_env().get()

    def test_gpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count > 0 for o in offers)

    # Offers in regions without a known provisioning host must not be advertised
    def test_locations_provisionable(self, offers: list[CatalogItem]) -> None:
        locations = {o.location for o in offers}
        assert locations <= set(JARVISLABS_REGION_URLS)

    # An unmapped GPU token would reach the catalog with its spaces intact
    def test_gpu_names_normalized(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            assert offer.gpu_name is None or " " not in offer.gpu_name, str(offer)

    def test_gpu_vendor_nvidia(self, offers: list[CatalogItem]) -> None:
        vendors = {o.gpu_vendor for o in offers if o.gpu_vendor}
        assert vendors == {AcceleratorVendor.NVIDIA}

    # JarvisLabs supports spot for containers and templates, not for the VMs dstack provisions
    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)
