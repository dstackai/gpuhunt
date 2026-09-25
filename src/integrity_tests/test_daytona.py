import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.daytona import GPU_MAP
from integrity_tests.base import CatalogFileIntegrityTests


class TestDaytonaCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "daytona"

    def test_no_unexpected_gpus(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {name for name, _, _, _ in GPU_MAP.values()}
        gpus = {o.gpu_name for o in offers if o.gpu_name}
        assert not gpus - expected_gpus

    @pytest.mark.parametrize("gpu_count", [1, 2, 4, 8])
    def test_gpu_count_present(self, gpu_count: int, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == gpu_count for o in offers)

    def test_both_capacity_types_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.spot for o in offers)
        assert any(not o.spot for o in offers)
