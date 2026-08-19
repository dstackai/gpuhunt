import logging
import os

import requests

from gpuhunt import CatalogItem, QueryFilter
from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers.base import OfflineProvider

logger = logging.getLogger(__name__)

CLOUDRIFT_SERVER_ADDRESS = "https://api.cloudrift.ai"
CLOUDRIFT_API_VERSION = "2025-03-21"


class CloudRiftProvider(OfflineProvider):
    NAME = "cloudrift"

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        instance_types = self._get_instance_types()
        offers = [offer for instance in instance_types for offer in _make_offers(instance)]
        return sorted(offers, key=lambda i: i.price)

    def _get_instance_types(self) -> list[dict]:
        request_data = {"selector": {"ByServiceAndLocation": {"services": ["vm"]}}}
        response_data = _make_request("instance-types/list", request_data)
        if not isinstance(response_data, dict):
            raise ValueError(f"Unexpected instance-types/list response: {response_data!r}")
        return response_data["instance_types"]


def _make_offers(instance: dict) -> list[CatalogItem]:
    instance_gpu_brand = instance["brand_short"]
    gpu_info = next(
        (gpu_record for gpu_record in GPU_MAP if gpu_record[0] in instance_gpu_brand), None
    )
    if gpu_info is None:
        logger.warning(f"Failed to find GPU name matching '{instance_gpu_brand}'")
        return []

    _, dstack_gpu_name, gpu_vendor = gpu_info
    instance_types = []
    for variant in instance["variants"]:
        for location, _count in variant["nodes_per_dc"].items():
            raw = CatalogItem(
                provider=CloudRiftProvider.NAME,
                instance_name=variant["name"],
                location=location,
                spot=False,
                price=variant["cost_per_hour"] / 100,
                cpu=variant["cpu_count"],
                memory=variant["dram"] / 1024**3,
                disk_size=variant["disk"] / 1024**3,
                gpu_count=variant["gpu_count"],
                gpu_name=dstack_gpu_name,
                gpu_memory=round(variant["vram"] / 1024**3),
                gpu_vendor=gpu_vendor,
            )
            instance_types.append(raw)

    return instance_types


GPU_MAP = [
    ("MI350X", "MI350X", AcceleratorVendor.AMD),
    ("RTX 4090", "RTX4090", AcceleratorVendor.NVIDIA),
    ("RTX 5090", "RTX5090", AcceleratorVendor.NVIDIA),
    ("RTX PRO 6000", "RTXPRO6000", AcceleratorVendor.NVIDIA),
]


def _make_request(endpoint: str, request_data: dict) -> dict | str:
    server = os.environ.get("CLOUDRIFT_SERVER_ADDRESS", CLOUDRIFT_SERVER_ADDRESS)
    response = requests.request(
        "POST",
        f"{server}/api/v1/{endpoint}",
        json={"version": CLOUDRIFT_API_VERSION, "data": request_data},
        timeout=5.0,
    )
    if not response.ok:
        response.raise_for_status()
    response_json = response.json()
    if isinstance(response_json, str):
        return response_json
    return response_json["data"]
