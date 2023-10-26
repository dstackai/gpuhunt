from pathlib import Path

import pytest


@pytest.fixture
def data(catalog_dir: Path) -> str:
    return (catalog_dir / "azure.csv").read_text()


class TestAzureCatalog:
    def test_standard_d2s_v3_locations(self, data: str):
        locations = [
            "attatlanta1",
            "attdallas1",
            "attdetroit1",
            "attnewyork1",
            "australiacentral",
            "australiacentral2",
            "australiaeast",
            "australiasoutheast",
            "australiasoutheast",
            "brazilsouth",
            "brazilsoutheast",
            "canadacentral",
            "canadaeast",
            "centralindia",
            "centralus",
            "eastasia",
            "eastus",
            "eastus2",
            "francecentral",
            "francesouth",
            "germanynorth",
            "germanywestcentral",
            "israelcentral",
            "italynorth",
            "japaneast",
            "japanwest",
            "jioindiacentral",
            "jioindiawest",
            "koreacentral",
            "koreasouth",
            "northcentralus",
            "northeurope",
            "norwayeast",
            "norwaywest",
            "polandcentral",
            "qatarcentral",
            "sgxsingapore1",
            "southafricanorth",
            "southafricawest",
            "southcentralus",
            "southeastasia",
            "southindia",
            "swedencentral",
            "swedensouth",
            "switzerlandnorth",
            "switzerlandwest",
            "uaecentral",
            "uaenorth",
            "uksouth",
            "ukwest",
            "usgovarizona",
            "usgovtexas",
            "usgovvirginia",
            "westcentralus",
            "westeurope",
            "westindia",
            "westus",
            "westus2",
            "westus3",
        ]
        assert all(f"\nStandard_D2s_v3,{i}," in data for i in locations)

    def test_spots_presented(self, data: str):
        assert ",True\n" in data

    def test_ondemand_presented(self, data: str):
        assert ",False\n" in data

    def test_gpu_presented(self, data: str):
        gpus = [
            "A100",
            "A10",
            "T4",
            "V100",
        ]
        assert all(f",{i}," in data for i in gpus)

    def test_both_a100_presented(self, data: str):
        assert ",A100,40.0," in data
        assert ",A100,80.0," in data
