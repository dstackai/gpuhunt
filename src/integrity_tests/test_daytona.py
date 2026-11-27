import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.daytona import DaytonaProvider, _fetch_pricing, _find_gpu
from integrity_tests.base import OffersIntegrityTests


@pytest.fixture(scope="module")
def feed_gpu_types() -> set[str]:
    return {gpu["type"] for gpu in _fetch_pricing()["gpus"]}


class TestDaytonaOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        provider = DaytonaProvider.from_env()
        offers = provider.get()
        if provider.api_key and not offers:
            pytest.skip("No Daytona offers are currently available")
        return offers

    def test_feed_gpu_types_are_known(self, feed_gpu_types: set[str]) -> None:
        # A new Daytona GPU is offered automatically once gpuhunt knows the accelerator.
        assert {t for t in feed_gpu_types if _find_gpu(t) is None} == set()

    def test_no_unexpected_gpus(self, offers: list[CatalogItem], feed_gpu_types: set[str]) -> None:
        expected_gpus = {gpu.name for t in feed_gpu_types if (gpu := _find_gpu(t))}
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

    def test_gpu_type_metadata(self, offers: list[CatalogItem], feed_gpu_types: set[str]) -> None:
        for offer in offers:
            if offer.gpu_count == 0:
                continue
            gpu_type = offer.provider_data.get("gpu_type", offer.gpu_name)
            assert isinstance(gpu_type, str)
            assert gpu_type in feed_gpu_types
            gpu = _find_gpu(gpu_type)
            assert gpu is not None and gpu.name == offer.gpu_name
            if gpu_type == offer.gpu_name:
                assert offer.provider_data == {}
            else:
                assert offer.provider_data == {"gpu_type": gpu_type}
