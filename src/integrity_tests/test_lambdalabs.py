import pytest

from gpuhunt import CatalogItem
from integrity_tests.base import CatalogFileIntegrityTests


class TestLambdaLabsCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "lambdalabs"

    @pytest.mark.parametrize("gpu", ["A10", "A100", "H100"])
    def test_gpu_present(self, gpu: str, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_name == gpu for o in offers)

    def test_on_demand_present(self, offers: list[CatalogItem]) -> None:
        assert any(not o.spot for o in offers)

    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)

    def test_locations(self, offers: list[CatalogItem]) -> None:
        expected_locations = {
            "asia-northeast-1",
            "asia-northeast-2",
            "asia-south-1",
            "australia-east-1",
            "europe-central-1",
            "me-west-1",
            "us-east-1",
            "us-east-3",
            "us-midwest-1",
            "us-south-2",
            "us-south-3",
            "us-west-1",
            "us-west-2",
            "us-west-3",
        }
        locations = {o.location for o in offers}
        assert not expected_locations - locations
