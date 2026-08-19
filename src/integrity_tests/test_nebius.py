from typing import cast

import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.nebius import NebiusCatalogItemProviderData
from integrity_tests.base import CatalogFileIntegrityTests


def get_offer(offers: list[CatalogItem], instance_name: str, location: str) -> CatalogItem:
    for offer in offers:
        if offer.instance_name == instance_name and offer.location == location:
            return offer
    raise LookupError(f"Offer not found: {instance_name} in {location}")


def get_fabrics(offer: CatalogItem) -> list[str]:
    return cast(NebiusCatalogItemProviderData, offer.provider_data)["fabrics"]


class TestNebiusCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "nebius"

    @pytest.mark.parametrize("gpu", ["RTXPRO6000", "L40S", "H100", "H200", "B200"])
    def test_gpu_present(self, gpu: str, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_name == gpu for o in offers)

    def test_cpu_offer_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == 0 for o in offers)

    def test_on_demand_present(self, offers: list[CatalogItem]) -> None:
        assert any(not o.spot for o in offers)

    def test_spot_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.spot for o in offers)

    @pytest.mark.parametrize("location", ["eu-north1", "eu-west1"])
    def test_location_present(self, location: str, offers: list[CatalogItem]) -> None:
        assert any(o.location == location for o in offers)

    def test_fabrics_unique(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            fabrics = get_fabrics(offer)
            assert len(fabrics) == len(set(fabrics)), str(offer)

    def test_fabrics_on_sample_offer(self, offers: list[CatalogItem]) -> None:
        offer = get_offer(offers, "gpu-h100-sxm 8gpu-128vcpu-1600gb", "eu-north1")
        expected_fabrics = {
            "fabric-2",
            "fabric-3",
            "fabric-4",
            "fabric-6",
        }
        assert not expected_fabrics - set(get_fabrics(offer))

    def test_no_fabrics_on_sample_non_clustered_offer(self, offers: list[CatalogItem]) -> None:
        offer = get_offer(offers, "gpu-h100-sxm 1gpu-16vcpu-200gb", "eu-north1")
        assert get_fabrics(offer) == []
