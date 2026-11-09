import warnings

import pytest
import requests

from gpuhunt import AcceleratorVendor, CatalogItem
from gpuhunt.providers.hydrahost import (
    GPU_MAP,
    INVENTORY_URL,
    ON_DEMAND_FILTER,
    TIMEOUT,
    HydraHostProvider,
)
from integrity_tests.base import OffersIntegrityTests


class TestHydraHostOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return HydraHostProvider.from_env().get()

    def test_gpu_vendor_nvidia(self, offers: list[CatalogItem]) -> None:
        assert {o.gpu_vendor for o in offers if o.gpu_count} <= {AcceleratorVendor.NVIDIA}

    def test_one_offer_per_server_and_contract(self, offers: list[CatalogItem]) -> None:
        keys = [(o.instance_name, o.spot) for o in offers]
        assert len(keys) == len(set(keys))

    def test_spot_cheaper_than_on_demand(self, offers: list[CatalogItem]) -> None:
        on_demand = {o.instance_name: o.price for o in offers if not o.spot}
        for o in offers:
            if o.spot and o.instance_name in on_demand:
                assert o.price <= on_demand[o.instance_name], str(o)

    def test_location_normalized(self, offers: list[CatalogItem]) -> None:
        assert all(o.location == o.location.lower() and " " not in o.location for o in offers)


def test_real_world_hydrahost_gpu_models():
    """Warn about live GPU models that would be skipped because they are not in GPU_MAP."""
    response = requests.get(
        INVENTORY_URL, params={"pageSize": 100, "filters": ON_DEMAND_FILTER}, timeout=TIMEOUT
    )
    response.raise_for_status()
    unknown = {
        str(gpu["model"])
        for device in response.json()["data"]
        if (gpu := (device.get("specs") or {}).get("gpu") or {}).get("count")
        and str(gpu.get("model")).strip().lower() not in GPU_MAP
    }
    if unknown:
        warnings.warn(
            f"Found {len(unknown)} Hydra Host GPU models without a mapping:\n"
            + "".join(f"- {name}\n" for name in sorted(unknown))
        )
