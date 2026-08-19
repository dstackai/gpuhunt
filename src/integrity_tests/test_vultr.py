import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.vultr import BARE_METAL_GPU_DETAILS, VultrProvider
from integrity_tests.base import OffersIntegrityTests


class TestVultrOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return VultrProvider.from_env().get()

    def test_gpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count > 0 for o in offers)

    def test_cpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == 0 for o in offers)

    def test_bare_metal_gpus_known(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {gpu_name for _, gpu_name, _memory in BARE_METAL_GPU_DETAILS.values()}
        bare_metal_gpus = {o.gpu_name for o in offers if o.instance_name.startswith("vbm-")}
        assert bare_metal_gpus - {None} <= expected_gpus

    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)
