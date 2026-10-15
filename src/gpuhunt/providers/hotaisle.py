import logging
from typing import TypedDict, cast

import requests
from requests import Response

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, JSONObject, QueryFilter
from gpuhunt.providers.base import OnlineProvider, get_creds_env

logger = logging.getLogger(__name__)

API_URL = "https://admin.hotaisle.app/api"
FLAG_BARE_METAL = "hotaisle-bm"


class HotAisleProvider(OnlineProvider):
    NAME = "hotaisle"

    def __init__(self, api_key: str, team_handle: str):
        """Hotaisle requries an API key and team handle to access the API."""
        self.api_key = api_key
        self.team_handle = team_handle

    @classmethod
    def from_env(cls) -> "HotAisleProvider":
        return cls(
            api_key=get_creds_env("HOTAISLE_API_KEY"),
            team_handle=get_creds_env("HOTAISLE_TEAM_HANDLE"),
        )

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        offers = self.fetch_offers()
        return sorted(offers, key=lambda i: i.price)

    def fetch_offers(self) -> list[CatalogItem]:
        """Fetch available virtual machines and bare metal servers from HotAisle API.
        See API documentation(https://admin.hotaisle.app/api/docs)
        for details.
        If one kind of offers fails, the error is logged and the other kind is still returned."""
        endpoints = [("virtual_machines", False), ("bare_metal", True)]
        offers: list[CatalogItem] = []
        errors: list[Exception] = []
        for endpoint, bare_metal in endpoints:
            try:
                response = self._make_request(
                    "GET", f"/teams/{self.team_handle}/{endpoint}/available/"
                )
                offers += _make_offers(response, bare_metal=bare_metal)
            except Exception as e:
                logger.exception("Failed to fetch Hot Aisle %s offers", endpoint)
                errors.append(e)
        if len(errors) == len(endpoints):
            raise errors[0]
        return offers

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


class HotAisleBareMetalCatalogItemProviderData(TypedDict):
    bare_metal_specs: JSONObject


def get_gpu_memory(gpu_name: str) -> float | None:
    if accelerators := find_accelerators(names=[gpu_name], vendors=[AcceleratorVendor.AMD]):
        return float(accelerators[0].memory)
    logger.warning(f"Unknown AMD GPU {gpu_name}")
    return None


def _make_offers(response: Response, bare_metal: bool) -> list[CatalogItem]:
    # The API returns null instead of an empty list when nothing is available.
    data = response.json() or []
    offers: list[CatalogItem] = []
    for item in data:
        price_in_cents = item["OnDemandPrice"]
        price = float(price_in_cents) / 100
        specs = item["Specs"]
        cpu_cores = specs["cpu_cores"]
        ram_capacity_bytes = specs["ram_capacity"]
        memory_gb = ram_capacity_bytes / (1024**3)
        disk_capacity_bytes = specs["disk_capacity"]
        disk_gb = disk_capacity_bytes / (1024**3)
        gpus = specs["gpus"]
        gpu = gpus[0]
        gpu_count = gpu["count"]
        gpu_name = gpu["model"]
        gpu_vendor = AcceleratorVendor.AMD  # All GPUs are AMD with HotAisle.
        gpu_memory = get_gpu_memory(gpu_name)

        kind = "vm"
        flags = []
        # The specs object may duplicate some CatalogItem fields, but we store it in
        # full because we need to pass it back to the API when creating instances.
        provider_data = cast(JSONObject, HotAisleCatalogItemProviderData(vm_specs=specs))
        if bare_metal:
            kind = "bm"
            flags.append(FLAG_BARE_METAL)
            provider_data = cast(
                JSONObject, HotAisleBareMetalCatalogItemProviderData(bare_metal_specs=specs)
            )
        # Create instance name: kind-gpu-gpucount, e.g. vm-mi300x-1 or bm-mi300x-8
        instance_name = f"{kind}-{gpu_name.lower()}-{gpu_count}"

        offer = CatalogItem(
            provider=HotAisleProvider.NAME,
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
            flags=flags,
            provider_data=provider_data,
        )
        offers.append(offer)

    return offers
