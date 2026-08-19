import logging
import os

import requests

from gpuhunt._internal.constraints import get_gpu_vendor
from gpuhunt._internal.models import CatalogItem, QueryFilter
from gpuhunt.providers.base import OnlineProvider, get_creds_env

logger = logging.getLogger(__name__)

# DigitalOcean Default API endpoints
STANDARD_CLOUD_API_URL = "https://api.digitalocean.com"


class DigitalOceanProvider(OnlineProvider):
    NAME = "digitalocean"

    def __init__(self, api_key: str, api_url: str = STANDARD_CLOUD_API_URL):
        self.api_key = api_key
        self.api_url = api_url

    @classmethod
    def from_env(cls) -> "DigitalOceanProvider":
        return cls(
            api_key=get_creds_env("DIGITAL_OCEAN_API_KEY"),
            api_url=os.getenv("DIGITAL_OCEAN_API_URL", STANDARD_CLOUD_API_URL),
        )

    def get(
        self, query_filter: QueryFilter | None = None, balance_resources: bool = True
    ) -> list[CatalogItem]:
        offers = self.fetch_offers()
        return sorted(offers, key=lambda i: i.price)

    def fetch_offers(self) -> list[CatalogItem]:
        url = "/v2/sizes"
        response = self._make_request("GET", url)
        return _make_offers(response)

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


def _make_offers(response) -> list[CatalogItem]:
    data = response.json()
    offers: list[CatalogItem] = []

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
            gpu_name = None
            gpu_memory = None

        total_disk_size = sum(
            float(disk["size"]["amount"]) for disk in size["disk_info"] if disk["type"] == "local"
        )

        memory_gb = float(size["memory"]) / 1024  # MB -> GB

        # Creates an offer for each available region.
        # If regions list is empty, instance type is not available.
        for region in size["regions"]:
            offer = CatalogItem(
                provider=DigitalOceanProvider.NAME,
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
