import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.cloudrift import GPU_MAP
from integrity_tests.base import CatalogFileIntegrityTests


class TestCloudRiftCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "cloudrift"

    def test_no_unexpected_gpus(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {gpu for _, gpu, _vendor in GPU_MAP}
        gpus = {o.gpu_name for o in offers if o.gpu_name}
        assert not gpus - expected_gpus

    # TODO: Add 3, 4, 5, ... 8
    @pytest.mark.parametrize("gpu_count", [1, 2])
    def test_gpu_count_present(self, gpu_count: int, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == gpu_count for o in offers)
