import logging
import os
from typing import Optional

import requests

from gpuhunt._internal.constraints import get_gpu_vendor
from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

# DigitalOcean Default API endpoints
STANDARD_CLOUD_API_URL = "https://api.digitalocean.com"


class DigitalOceanProvider(AbstractProvider):
    NAME = "digitalocean"

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("DIGITAL_OCEAN_API_KEY")
        if not self.api_key:
            raise ValueError("Set the DIGITAL_OCEAN_API_KEY environment variable.")

        self.api_url = api_url or os.getenv("DIGITAL_OCEAN_API_URL", STANDARD_CLOUD_API_URL)

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        offers = self.fetch_offers()
        return sorted(offers, key=lambda i: i.price)

    def fetch_offers(self) -> list[RawCatalogItem]:
        url = "/v2/sizes"
        response = self._make_request("GET", url)
        return convert_response_to_raw_catalog_items(response)

    def _make_request(self, method: str, url: str):
        full_url = f"{self.api_url}{url}"
        params = {"per_page": 500}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.request(
            method=method, url=full_url, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()
        return response


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
            gpu_name = "".join(part.upper() for part in model_parts[1:])
            gpu_vendor = get_gpu_vendor(gpu_name)
            if gpu_vendor is None:
                logger.warning(
                    f"Could not determine GPU vendor for model '{gpu_model}'. Skipping droplet '{size['slug']}'."
                )
                continue
        else:
            gpu_count = 0
            gpu_vendor = None
            gpu_name = ""
            gpu_memory = 0

        total_disk_size = sum(
            float(disk["size"]["amount"]) for disk in size["disk_info"] if disk["type"] == "local"
        )

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
