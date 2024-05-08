from pathlib import Path

import pytest


@pytest.fixture
def data(catalog_dir: Path) -> str:
    return (catalog_dir / "nebius.csv").read_text()


class TestNebiusCatalog:
    @pytest.mark.xfail
    def test_zone_presented(self, data: str):
        assert ",eu-north1-c," in data

    @pytest.mark.xfail
    def test_gpu_presented(self, data: str):
        gpus = [
            "A100",
            "L4",
            "L40",
            "H100",
        ]
        assert all(f",{i}," in data for i in gpus)

    @pytest.mark.xfail
    def test_h100_platforms(self, data: str):
        platforms = [
            "gpu-h100",
            "gpu-h100-b",
            "standard-v3-h100-pcie",
        ]
        assert all(f"\n{i}," in data for i in platforms)

    @pytest.mark.xfail
    def test_no_spots(self, data: str):
        assert ",True\n" not in data
