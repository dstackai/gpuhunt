from collections import Counter

from gpuhunt import CatalogItem
from gpuhunt.providers.verda import ALL_AMD_GPUS, GPU_MAP
from integrity_tests.base import CatalogFileIntegrityTests


class TestVerdaCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "verda"

    def test_locations(self, offers: list[CatalogItem]) -> None:
        expected_locations = {
            "FIN-01",
            "FIN-02",
        }
        locations = Counter(o.location for o in offers)
        assert not expected_locations - set(locations)
        for location in expected_locations:
            assert locations[location] > 1

    def test_spot_and_on_demand_present(self, offers: list[CatalogItem]) -> None:
        spots = Counter(o.spot for o in offers)
        assert spots[True] > 1
        assert spots[False] > 1

    def test_gpus(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {name for name in GPU_MAP.values() if name not in ALL_AMD_GPUS}
        gpus = {o.gpu_name for o in offers if o.gpu_name}
        assert gpus == expected_gpus
