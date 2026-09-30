import pytest

from gpuhunt import AcceleratorVendor, CatalogItem
from integrity_tests.base import CatalogFileIntegrityTests


class TestOCICatalog(CatalogFileIntegrityTests):
    CATALOG_NAME = "oci"

    @pytest.mark.parametrize(
        "gpu",
        [
            "P100",
            "V100",
            "A10",
            "A100",
            "L40S",
            "H100",
            "H200",
            "MI300X",
            "MI355X",
            "B200",
            "B300",
            "GB200",
            "GB300",
            "RTXPRO6000",
        ],
    )
    def test_gpu_present(self, gpu: str, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_name == gpu for o in offers)

    def test_amd_offers_have_amd_vendor(self, offers: list[CatalogItem]) -> None:
        amd_offers = [o for o in offers if o.gpu_name in ("MI300X", "MI355X")]
        assert amd_offers
        assert all(o.gpu_vendor == AcceleratorVendor.AMD for o in amd_offers)

    def test_shape_names_match_compute_api(self, offers: list[CatalogItem]) -> None:
        # The Cost Estimator names some shapes "BM.GPU.GB200.4 (NVL72)"
        assert not any("(" in o.instance_name for o in offers)

    def test_cpu_offer_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count == 0 for o in offers)

    def test_on_demand_present(self, offers: list[CatalogItem]) -> None:
        assert any(not o.spot for o in offers)

    def test_spot_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.spot for o in offers)

    def test_spots_contain_flag(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            assert offer.spot == ("oci-spot" in offer.flags), str(offer)

    @pytest.mark.parametrize("prefix", ["VM.Standard", "BM.Standard", "VM.GPU", "BM.GPU"])
    def test_family_present(self, prefix: str, offers: list[CatalogItem]) -> None:
        assert any(o.instance_name.startswith(prefix) for o in offers)

    def test_quantity_decreases_as_query_complexity_increases(
        self, offers: list[CatalogItem]
    ) -> None:
        zero_or_one_gpu = [o for o in offers if o.gpu_count in (0, 1)]
        zero_gpu = [o for o in offers if o.gpu_count == 0]
        one_gpu = [o for o in offers if o.gpu_count == 1]

        assert len(offers) > len(zero_or_one_gpu)
        assert len(zero_or_one_gpu) > len(zero_gpu)
        assert len(zero_gpu) > len(one_gpu)
