import warnings

import pytest
import requests

from gpuhunt import AcceleratorVendor, CatalogItem
from gpuhunt._internal.constraints import find_accelerators
from gpuhunt.providers.base import get_creds_env
from gpuhunt.providers.runcrate import GPU_MAP, INSTANCE_TYPES_URL, TIMEOUT, RuncrateProvider
from integrity_tests.base import OffersIntegrityTests


class TestRuncrateOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return RuncrateProvider.from_env().get()

    def test_gpu_offers_present(self, offers: list[CatalogItem]) -> None:
        assert any(o.gpu_count > 0 for o in offers)

    def test_gpu_vendor_nvidia(self, offers: list[CatalogItem]) -> None:
        vendors = {o.gpu_vendor for o in offers if o.gpu_count > 0}
        assert vendors == {AcceleratorVendor.NVIDIA}

    # Unknown types are skipped, so every published name must be a known GPU
    def test_gpu_names_known(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            if offer.gpu_count > 0:
                assert offer.gpu_name and find_accelerators(names=[offer.gpu_name]), str(offer)

    def test_disk_size_present(self, offers: list[CatalogItem]) -> None:
        assert all(o.disk_size for o in offers)

    def test_spot_not_present(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)


def test_real_world_runcrate_gpu_types():
    """Warn about live ``gpu_type`` values that would be skipped as unknown GPUs."""
    response = requests.get(
        INSTANCE_TYPES_URL,
        headers={"Authorization": f"Bearer {get_creds_env('RUNCRATE_API_KEY')}"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    unknown = {
        t["gpu_type"]
        for t in response.json()["data"]
        if t.get("gpu_count") and t.get("gpu_type") not in GPU_MAP
    }
    if unknown:
        warnings.warn(
            f"Found {len(unknown)} Runcrate GPU types without a known GPU name:\n"
            + "".join(f"- {name}\n" for name in sorted(unknown))
        )
