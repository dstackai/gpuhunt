from gpuhunt import CatalogItem
from integrity_tests.base import CatalogFileIntegrityTests


class TestAzureCatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "azure"

    def test_standard_d2s_v3_locations(self, offers: list[CatalogItem]) -> None:
        expected_locations = {
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
            "indonesiacentral",
            "israelcentral",
            "italynorth",
            "japaneast",
            "japanwest",
            "jioindiacentral",
            "jioindiawest",
            "koreacentral",
            "koreasouth",
            "malaysiawest",
            "mexicocentral",
            "newzealandnorth",
            "northcentralus",
            "northeurope",
            "norwayeast",
            "norwaywest",
            "polandcentral",
            "qatarcentral",
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
            "westcentralus",
            "westeurope",
            "westindia",
            "westus",
            "westus2",
            "westus3",
        }
        locations = {o.location for o in offers if o.instance_name == "Standard_D2s_v3"}
        assert not expected_locations - locations

    def test_spot_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.spot for o in offers)

    def test_on_demand_present(self, offers: list[CatalogItem]) -> None:
        assert any(not o.spot for o in offers)

    def test_gpu_present(self, offers: list[CatalogItem]) -> None:
        expected_gpus = {
            "A100",
            "A10",
            "H100NVL",
            "H200",
            "T4",
            "V100",
        }
        gpus = {o.gpu_name for o in offers if o.gpu_name}
        assert gpus == expected_gpus

    def test_both_a100_present(self, offers: list[CatalogItem]) -> None:
        gpu_memory = {o.gpu_memory for o in offers if o.gpu_name == "A100"}
        assert gpu_memory == {40.0, 80.0}
