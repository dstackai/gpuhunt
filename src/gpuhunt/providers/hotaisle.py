import logging
import os
from typing import Optional, TypedDict, cast

import requests
from requests import Response

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import AcceleratorVendor, JSONObject, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

API_URL = "https://admin.hotaisle.app/api"


class HotAisleProvider(AbstractProvider):
    NAME = "hotaisle"

    def __init__(self, api_key: Optional[str] = None, team_handle: Optional[str] = None):
        """Hotaisle requries an API key and team handle to access the API."""
        self.api_key = api_key or os.getenv("HOTAISLE_API_KEY")
        self.team_handle = team_handle or os.getenv("HOTAISLE_TEAM_HANDLE")

        if not self.api_key:
            raise ValueError("Set the HOTAISLE_API_KEY environment variable.")
        if not self.team_handle:
            raise ValueError("Set the HOTAISLE_TEAM_HANDLE environment variable.")

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        offers = self.fetch_offers()
        return sorted(offers, key=lambda i: i.price)

    def fetch_offers(self) -> list[RawCatalogItem]:
        """Fetch available virtual machines from HotAisle API.
        See API documentation(https://admin.hotaisle.app/api/docs)
        for details."""
        url = f"/teams/{self.team_handle}/virtual_machines/available/"
        response = self._make_request("GET", url)
        return convert_response_to_raw_catalog_items(response)

    def _make_request(self, method: str, url: str) -> Response:
        full_url = f"{API_URL}{url}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

        response = requests.request(method=method, url=full_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response


class HotAisleCatalogItemProviderData(TypedDict):
    vm_specs: JSONObject


def get_gpu_memory(gpu_name: str) -> Optional[float]:
    if accelerators := find_accelerators(names=[gpu_name], vendors=[AcceleratorVendor.AMD]):
        return float(accelerators[0].memory)
    logger.warning(f"Unknown AMD GPU {gpu_name}")
    return None


def convert_response_to_raw_catalog_items(response: Response) -> list[RawCatalogItem]:
    data = response.json()
    offers = []
    for item in data:
        price_in_cents = item["OnDemandPrice"]
        price = float(price_in_cents) / 100
        specs = item["Specs"]
        cpu_cores = specs["cpu_cores"]
        ram_capacity_bytes = specs["ram_capacity"]
        memory_gb = ram_capacity_bytes / (1024**3)
        disk_capacity_bytes = specs["disk_capacity"]
        disk_gb = disk_capacity_bytes / (1024**3)
        cpus = specs["cpus"]
        cpu_model = cpus["model"]
        gpus = specs["gpus"]
        gpu = gpus[0]
        gpu_count = gpu["count"]
        gpu_name = gpu["model"]
        gpu_vendor = AcceleratorVendor.AMD.value  # All GPUs are AMD with HotAisle.
        gpu_memory = get_gpu_memory(gpu_name)

        # Create instance name: cpu_model-cores-ram-gpucount-gpu
        instance_name = f"{gpu_count}x {gpu_name} {cpu_cores}x {cpu_model}"

        offer = RawCatalogItem(
            instance_name=instance_name,
            location="us-michigan-1",  # Hardcoded for now, as HotAisle only has one location.
            price=price,
            cpu=cpu_cores,
            memory=memory_gb,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_memory=gpu_memory,
            gpu_vendor=gpu_vendor,
            spot=False,
            disk_size=disk_gb,
            provider_data=cast(
                JSONObject,
                HotAisleCatalogItemProviderData(
                    # The specs object may duplicate some RawCatalogItem fields, but we store it in
                    # full because we need to pass it back to the API when creating VMs.
                    vm_specs=specs,
                ),
            ),
        )
        offers.append(offer)

    return offers
