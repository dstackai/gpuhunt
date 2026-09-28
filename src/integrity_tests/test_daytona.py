import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.daytona import GPU_MAP, DaytonaProvider
from integrity_tests.base import OffersIntegrityTests


class TestDaytonaOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        provider = DaytonaProvider.from_env()
        offers = provider.get()
        if provider.api_key and not offers:
            pytest.skip("No Daytona offers are currently available")
        return offers

    def test_no_unexpected_gpus(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {info[0] for info in GPU_MAP.values()}
        gpus = {o.gpu_name for o in offers if o.gpu_name}
        assert not gpus - expected_gpus

    def test_supported_gpu_counts(self, offers: list[CatalogItem]) -> None:
        assert all(0 <= o.gpu_count <= 8 for o in offers)

    def test_shared_region(self, offers: list[CatalogItem]) -> None:
        assert all(o.location == "earth" for o in offers if o.gpu_count)
        assert all(o.location and o.location != "earth" for o in offers if not o.gpu_count)

    def test_cpu_metadata(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            if offer.gpu_count == 0:
                assert offer.provider_data == {}
                assert not offer.spot

    def test_gpu_type_metadata(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            if offer.gpu_count == 0:
                continue
            gpu_type = offer.provider_data.get("gpu_type", offer.gpu_name)
            assert isinstance(gpu_type, str)
            assert gpu_type in GPU_MAP
            assert GPU_MAP[gpu_type][0] == offer.gpu_name
            if gpu_type == offer.gpu_name:
                assert offer.provider_data == {}
            else:
                assert offer.provider_data == {"gpu_type": gpu_type}
