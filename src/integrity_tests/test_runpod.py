import re

from gpuhunt import CatalogItem
from gpuhunt.providers.runpod import get_gpu_map
from integrity_tests.base import CatalogFileIntegrityTests


class TestRunpodCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "runpod"

    def test_locations(self, offers: list[CatalogItem]) -> None:
        locations = {o.location for o in offers}
        assert 10 <= len(locations) <= 300
        # Secure cloud locations are datacenter IDs, e.g., EU-RO-1
        secure_cloud = {loc for loc in locations if re.fullmatch(r"[A-Z]+-[A-Z]+-\d+", loc)}
        # Community cloud locations are country codes, e.g., FR
        community_cloud = {loc for loc in locations if re.fullmatch(r"[A-Z]{2}", loc)}
        assert secure_cloud
        assert community_cloud

    def test_gpu_present(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {name for _, name in get_gpu_map().values()}
        gpus = {o.gpu_name for o in offers if o.gpu_name}
        assert len(expected_gpus & gpus) > 7

    def test_cpu_offers_integrity(self, offers: list[CatalogItem]) -> None:
        cpu_offers = [o for o in offers if o.gpu_count == 0]
        assert cpu_offers
        for offer in cpu_offers:
            assert "runpod-cpu" in offer.flags, str(offer)
            assert not offer.spot, str(offer)
            assert "-" in offer.location, str(offer)
