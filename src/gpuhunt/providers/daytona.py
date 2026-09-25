import logging
from typing import Any

import requests

from gpuhunt import CatalogItem, QueryFilter
from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers.base import OfflineProvider

logger = logging.getLogger(__name__)

# Public rate card, no authentication. Served from the same definitions the
# billing rate card is built from; both capacity types (on-demand and spot)
# are published for GPUs and resources alike.
API_URL = "https://billing.app.daytona.io/gpu-pricing"
TIMEOUT = 10.0

# Daytona bills GPU sandboxes a la carte: each GPU type carries a per-GPU
# hourly rate, and vCPU/RAM/disk are billed separately per second, so there
# are no fixed instance types. The catalog therefore publishes representative
# sandbox configurations: per GPU, 8 vCPU with 100 GB RAM on datacenter-class
# cards (50 GB on consumer cards) and 256 GB disk, scaled linearly with the
# GPU count. Spot offers price the GPU and the resources at Daytona's spot
# rates, which are discounted independently of the on-demand rates.
GPU_COUNTS = [1, 2, 4, 8]
CPUS_PER_GPU = 8
DISK_PER_GPU = 256.0  # GB
MEMORY_PER_GPU = 100.0  # GB, datacenter-class cards
CONSUMER_MEMORY_PER_GPU = 50.0  # GB, consumer-class cards

# Daytona GPU type -> (canonical gpuhunt name, vendor, VRAM in GB, RAM per GPU in GB)
GPU_MAP: dict[str, tuple[str, AcceleratorVendor, float, float]] = {
    "B300": ("B300", AcceleratorVendor.NVIDIA, 270.0, MEMORY_PER_GPU),
    "H100": ("H100", AcceleratorVendor.NVIDIA, 80.0, MEMORY_PER_GPU),
    "H200": ("H200", AcceleratorVendor.NVIDIA, 141.0, MEMORY_PER_GPU),
    "B200": ("B200", AcceleratorVendor.NVIDIA, 180.0, MEMORY_PER_GPU),
    "MI355X": ("MI355X", AcceleratorVendor.AMD, 288.0, MEMORY_PER_GPU),
    "RTX-PRO-6000": ("RTXPRO6000", AcceleratorVendor.NVIDIA, 96.0, MEMORY_PER_GPU),
    "RTX-5090": ("RTX5090", AcceleratorVendor.NVIDIA, 32.0, CONSUMER_MEMORY_PER_GPU),
    "RTX-4090": ("RTX4090", AcceleratorVendor.NVIDIA, 24.0, CONSUMER_MEMORY_PER_GPU),
}

# GPU capacity is pooled and scheduled across Daytona's fleet; a specific
# region is not user-selectable on shared capacity, so offers are published
# under the primary region.
LOCATION = "us"


class DaytonaProvider(OfflineProvider):
    NAME = "daytona"

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        pricing = _fetch_pricing()
        offers = _make_offers(pricing)
        return sorted(offers, key=lambda i: i.price)


def _fetch_pricing() -> dict[str, Any]:
    response = requests.get(API_URL, timeout=TIMEOUT)
    response.raise_for_status()
    pricing = response.json()
    if (
        not isinstance(pricing, dict)
        or not isinstance(pricing.get("gpus"), list)
        or not isinstance(pricing.get("resources"), dict)
        or not isinstance(pricing["resources"].get("onDemand"), dict)
        or not isinstance(pricing["resources"].get("spot"), dict)
    ):
        raise ValueError(f"Unexpected gpu-pricing response: {pricing!r}")
    return pricing


def _make_offers(pricing: dict[str, Any]) -> list[CatalogItem]:
    offers = []
    for gpu in pricing["gpus"]:
        gpu_info = GPU_MAP.get(gpu["type"])
        if gpu_info is None:
            logger.warning("Failed to find GPU name matching '%s'", gpu["type"])
            continue
        gpu_name, gpu_vendor, gpu_memory, memory_per_gpu = gpu_info
        for spot in (False, True):
            gpu_rate = gpu["spotPricePerHour" if spot else "onDemandPricePerHour"]
            resources = pricing["resources"]["spot" if spot else "onDemand"]
            for gpu_count in GPU_COUNTS:
                cpu = CPUS_PER_GPU * gpu_count
                memory = memory_per_gpu * gpu_count
                disk_size = DISK_PER_GPU * gpu_count
                price = round(
                    gpu_count * gpu_rate
                    + cpu * resources["vcpuPerHour"]
                    + memory * resources["memoryGiBPerHour"]
                    + disk_size * resources["diskGiBPerHour"],
                    6,
                )
                offers.append(
                    CatalogItem(
                        provider=DaytonaProvider.NAME,
                        instance_name=f"{gpu_count}x-{gpu_name}",
                        location=LOCATION,
                        price=price,
                        cpu=cpu,
                        memory=memory,
                        gpu_count=gpu_count,
                        gpu_name=gpu_name,
                        gpu_memory=gpu_memory,
                        spot=spot,
                        disk_size=disk_size,
                        gpu_vendor=gpu_vendor,
                    )
                )
    return offers
