from gpuhunt import CatalogItem
from integrity_tests.base import CatalogFileIntegrityTests


class TestAWSCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "aws"

    def test_m5_large_locations(self, offers: list[CatalogItem]) -> None:
        expected_locations = {
            "af-south-1",
            "ap-east-1",
            "ap-northeast-1",
            "ap-northeast-2",
            "ap-northeast-3",
            "ap-south-1",
            "ap-south-2",
            "ap-southeast-1",
            "ap-southeast-2",
            "ap-southeast-3",
            "ap-southeast-4",
            "ca-central-1",
            "eu-central-1",
            "eu-central-2",
            "eu-north-1",
            "eu-south-1",
            "eu-south-2",
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "il-central-1",
            "me-central-1",
            "me-south-1",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-gov-east-1",
            "us-gov-west-1",
            "us-west-1",
            "us-west-2",
            "us-west-2-lax-1",
        }
        locations = {o.location for o in offers if o.instance_name == "m5.large"}
        assert not expected_locations - locations

    def test_spot_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.spot for o in offers)

    def test_on_demand_present(self, offers: list[CatalogItem]) -> None:
        assert any(not o.spot for o in offers)

    def test_gpu_present(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {
            # AWS pricing csv does not include H200 (p5e.) offers.
            # TODO: Add CapacityBlocks offers to support H200.
            # "H200",
            "H100",
            "A100",
            "A10G",
            "T4",
            "L40S",
            "RTXPRO4500",
            "RTXPRO6000",
            "L4",
        }
        gpus = {o.gpu_name for o in offers}
        assert not expected_gpus - gpus
