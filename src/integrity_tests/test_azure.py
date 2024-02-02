import csv
from pathlib import Path
from typing import List

import pytest


@pytest.fixture
def data_rows(catalog_dir: Path) -> List[dict]:
    with open(catalog_dir / "azure.csv", "r") as f:
        return list(csv.DictReader(f))


class TestAzureCatalog:
    def test_standard_d2s_v3_locations(self, data_rows: List[dict]):
        expected_locations = {
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
            "mexicocentral",
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
            "spaincentral",
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
        }
        locations = set(
            row["location"] for row in data_rows if row["instance_name"] == "Standard_D2s_v3"
        )
        assert expected_locations == locations

    def test_spots_presented(self, data_rows: List[dict]):
        assert any(row["spot"] == "True" for row in data_rows)

    def test_ondemand_presented(self, data_rows: List[dict]):
        assert any(row["spot"] == "False" for row in data_rows)

    def test_gpu_presented(self, data_rows: List[dict]):
        expected_gpus = {
            "A100",
            "A10",
            "T4",
            "V100",
        }
        gpus = set(row["gpu_name"] for row in data_rows if row["gpu_name"])
        assert expected_gpus == gpus

    def test_both_a100_presented(self, data_rows: List[dict]):
        expected_gpu_memory = {"40.0", "80.0"}
        gpu_memory = set(row["gpu_memory"] for row in data_rows if row["gpu_name"] == "A100")
        assert expected_gpu_memory == gpu_memory
