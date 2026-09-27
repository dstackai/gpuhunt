import warnings

import pytest
import requests

from gpuhunt import AcceleratorVendor, CatalogItem
from gpuhunt._internal.constraints import find_accelerators
from gpuhunt.providers.lium import NODES_URL, TIMEOUT, LiumProvider, get_dstack_gpu_name
from integrity_tests.base import OffersIntegrityTests


class TestLiumOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return LiumProvider.from_env().get()

    # Lium rents GPU pods only
    def test_all_offers_have_gpus(self, offers: list[CatalogItem]) -> None:
        assert all(o.gpu_count > 0 for o in offers)

    def test_gpu_vendor_nvidia(self, offers: list[CatalogItem]) -> None:
        vendors = {o.gpu_vendor for o in offers}
        assert vendors == {AcceleratorVendor.NVIDIA}

    # Unmapped models are skipped, so every published name must be a known GPU
    def test_gpu_names_known(self, offers: list[CatalogItem]) -> None:
        for offer in offers:
            assert offer.gpu_name and find_accelerators(names=[offer.gpu_name]), str(offer)

    def test_price_is_per_gpu_times_count(self, offers: list[CatalogItem]) -> None:
        # Offers for the same node differ only in GPU count, so price must scale linearly.
        by_node: dict[str, list[CatalogItem]] = {}
        for o in offers:
            by_node.setdefault(o.instance_name, []).append(o)
        for node_offers in by_node.values():
            per_gpu = node_offers[0].price / node_offers[0].gpu_count
            for o in node_offers:
                assert o.price == pytest.approx(per_gpu * o.gpu_count, abs=1e-4), node_offers

    def test_location_present(self, offers: list[CatalogItem]) -> None:
        assert all(o.location and o.location == o.location.lower() for o in offers)


def test_real_world_lium_gpu_models():
    """Warn about live ``gpu_model`` values that would be skipped as unknown GPUs."""
    response = requests.get(NODES_URL, timeout=TIMEOUT)
    response.raise_for_status()
    unknown = set()
    for node in response.json()["nodes"]:
        gpu_model = node.get("gpu_model")
        if not gpu_model:
            continue
        gpu_name = get_dstack_gpu_name(gpu_model)
        if not find_accelerators(names=[gpu_name], vendors=[AcceleratorVendor.NVIDIA]):
            unknown.add(gpu_model)
    if unknown:
        warnings.warn(
            f"Found {len(unknown)} Lium GPU models without a known GPU name:\n"
            + "".join(f"- {name}\n" for name in sorted(unknown))
        )
