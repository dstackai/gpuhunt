import logging
import os
from typing import Optional

import requests

from gpuhunt._internal.constraints import KNOWN_AMD_GPUS, KNOWN_NVIDIA_GPUS
from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

# DigitalOcean API endpoints
STANDARD_CLOUD_API_URL = "https://api.digitalocean.com"
AMD_CLOUD_API_URL = "https://api-amd.digitalocean.com"


class DigitalOceanProvider(AbstractProvider):
    NAME = "digitalocean"

    def __init__(self, token: Optional[str] = None, flavor: Optional[str] = None):
        self.token = token or os.getenv("DIGITAL_OCEAN_TOKEN")
        if not self.token:
            raise ValueError("Set the DIGITAL_OCEAN_TOKEN environment variable.")

        flavor = flavor or os.getenv("DIGITAL_OCEAN_FLAVOR", "standard").lower()
        if flavor not in ("amd", "standard"):
            flavor = "standard"
        self.api_url = AMD_CLOUD_API_URL if flavor == "amd" else STANDARD_CLOUD_API_URL

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        offers = self.fetch_offers()
        return sorted(offers, key=lambda i: (i.price is None, i.price))

    def fetch_offers(self) -> list[RawCatalogItem]:
        url = "/v2/sizes"
        response = self._make_request("GET", url)
        return convert_response_to_raw_catalog_items(response)

    def _make_request(self, method: str, url: str):
        full_url = f"{self.api_url}{url}"
        params = {"per_page": 500}
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.request(
            method=method, url=full_url, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()
        return response


def get_gpu_vendor(gpu_name: Optional[str]) -> Optional[str]:
    if gpu_name is None:
        return None
    for gpu in KNOWN_NVIDIA_GPUS:
        if gpu.name.upper() == gpu_name.upper():
            return AcceleratorVendor.NVIDIA.value
    for gpu in KNOWN_AMD_GPUS:
        if gpu.name.upper() == gpu_name.upper():
            return AcceleratorVendor.AMD.value
    return None


def convert_response_to_raw_catalog_items(response) -> list[RawCatalogItem]:
    data = response.json()
    offers = []

    for size in data["sizes"]:
        gpu_info = size.get("gpu_info")
        if gpu_info:
            gpu_count = gpu_info["count"]
            gpu_vram_info = gpu_info["vram"]
            gpu_memory = float(gpu_vram_info["amount"])
            gpu_model = gpu_info["model"]
            # gpu_model uses patterns like "amd_mi300x", "nvidia_h100", "nvidia_rtx6000_ada"
            model_parts = gpu_model.split("_")
            if len(model_parts) >= 3:
                # Handle cases like "nvidia_rtx6000_ada" -> "RTX6000ADA"
                gpu_name = "".join(part.upper() for part in model_parts[1:])
            else:
                # Handle cases like "amd_mi300x" -> "MI300X"
                gpu_name = model_parts[1].upper()
            gpu_vendor = get_gpu_vendor(gpu_name)
        else:
            gpu_count = 0
            gpu_vendor = None
            gpu_name = ""
            gpu_memory = 0
            gpu_model = ""

        # Aggregate disk sizes (local and scratch).
        total_disk_size = 0.0
        for disk in size["disk_info"]:
            total_disk_size += float(disk["size"]["amount"])

        memory_gb = float(size["memory"]) / 1024  # MB -> GB

        # Creates an offer for each available region.
        # If regions list is empty, instance type is not available.
        for region in size["regions"]:
            offer = RawCatalogItem(
                instance_name=size["slug"],
                location=region,
                price=size["price_hourly"],
                cpu=size["vcpus"],
                memory=memory_gb,
                gpu_vendor=gpu_vendor,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                spot=False,
                disk_size=total_disk_size,
                flags=[],
            )
            offers.append(offer)

    return offers
