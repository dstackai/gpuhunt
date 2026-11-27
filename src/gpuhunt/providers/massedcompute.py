import logging
import re
from typing import TypedDict

import requests

from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, QueryFilter
from gpuhunt.providers.base import OnlineProvider, get_creds_env

logger = logging.getLogger(__name__)

INVENTORY_URL = "https://vm.massedcompute.com/api/v1/gpu-inventory"

# Model labels in GET /gpu-inventory. Configurations can differ in host RAM,
# local disk and interconnect while sharing the same GPU model.
_GPU_NAMES = {
    "RTX A5000": "A5000",
    "RTX A6000": "A6000",
    "RTX 6000 ADA": "RTX6000Ada",
    "RTX PRO 4500 Blackwell": "RTXPRO4500",
    "RTX PRO 6000 Blackwell": "RTXPRO6000",
    "A30": "A30",
    "L40": "L40",
    "L40S": "L40S",
    "A100": "A100",
    "A100 SXM4": "A100",
    "DGX A100": "A100",
    "H100": "H100",
    "H100 SXM5": "H100",
    "H100 NVL": "H100NVL",
    "H200 NVL": "H200NVL",
    "B200 SXM6": "B200",
    "B300 SXM6": "B300",
}


class _Specs(TypedDict):
    vcpu_count: int
    memory_gib: int
    storage_gb: int


class _InstanceType(TypedDict):
    name: str
    description: str
    price_cents_per_hour: int
    specs: _Specs


class _InventoryItem(TypedDict):
    instance_type: _InstanceType
    capacity_available: int


class MassedComputeProvider(OnlineProvider):
    """On-demand VMs. The launch API accepts a product name and places it at any site."""

    NAME = "massedcompute"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "MassedComputeProvider":
        return cls(get_creds_env("MASSEDCOMPUTE_API_KEY"))

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        response = requests.get(
            INVENTORY_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        inventory: dict[str, _InventoryItem] = response.json()["gpu_inventory"]
        offers = []
        for item in inventory.values():
            product = item["instance_type"]
            if item["capacity_available"] <= 0 or product["price_cents_per_hour"] <= 0:
                continue
            if offer := _make_offer(product):
                offers.append(offer)
        return sorted(offers, key=lambda offer: offer.price)


def _make_offer(product: _InstanceType) -> CatalogItem | None:
    name = product["name"]
    gpu_count, gpu_name, gpu_memory = 0, None, None
    if name.startswith("gpu_"):
        match = re.fullmatch(
            r"(\d+)x (.+?)(?: \((\d+)GB\))?(?: NVLink)?(?: \[[^\]]+\])?", product["description"]
        )
        if match is None or (gpu_name := _GPU_NAMES.get(match[2])) is None:
            logger.warning("Skipping unknown MassedCompute GPU product %s", name)
            return None
        gpu_count = int(match[1])
        if match[3]:
            gpu_memory = int(match[3])
        elif gpu_name == "H100NVL":
            # The inventory also lists this exact board as "H100 NVL (94GB) NVLink".
            gpu_memory = 94
        else:
            logger.warning("Skipping MassedCompute product %s without GPU memory", name)
            return None
    elif not (name.startswith("cpu_") and product["description"].endswith("AMD EPYC")):
        logger.warning("Skipping unknown MassedCompute product %s", name)
        return None
    specs = product["specs"]
    return CatalogItem(
        provider=MassedComputeProvider.NAME,
        instance_name=name,
        # Capacity is aggregated across sites. The launch API documents only "any".
        location="any",
        price=product["price_cents_per_hour"] / 100,
        cpu=specs["vcpu_count"],
        memory=specs["memory_gib"],
        disk_size=specs["storage_gb"],
        gpu_count=gpu_count,
        gpu_vendor=AcceleratorVendor.NVIDIA if gpu_count else None,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        spot=False,
    )
