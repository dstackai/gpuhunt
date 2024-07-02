from pathlib import Path

import pytest


@pytest.fixture
def data(catalog_dir: Path) -> str:
    return (catalog_dir / "aws.csv").read_text()


class TestAWSCatalog:
    def test_m5_large_regions(self, data: str):
        instance = "m5.large"
        regions = [
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
        ]
        assert all(f"\n{instance},{i}," in data for i in regions)

    def test_spots_presented(self, data: str):
        assert ",True," in data

    def test_gpu_presented(self, data: str):
        gpus = [
            "H100",
            "A100",
            "A10G",
            "T4",
            "V100",
            "L4",
        ]
        assert all(f",{i}," in data for i in gpus)
