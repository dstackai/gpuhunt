import logging
from typing import Optional, Union

import requests

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

CLOUDRIFT_SERVER_ADDRESS = "https://api.cloudrift.ai"
CLOUDRIFT_API_VERSION = "2025-03-21"


class CloudRiftProvider(AbstractProvider):
    NAME = "cloudrift"

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        instance_types = self._get_instance_types()
        instance_types = [
            inst for instance in instance_types for inst in generate_instances(instance)
        ]
        return sorted(instance_types, key=lambda x: x.price)

    def _get_instance_types(self):
        request_data = {"selector": {"ByServiceAndLocation": {"services": ["vm"]}}}
        response_data = _make_request("instance-types/list", request_data)
        return response_data["instance_types"]


def generate_instances(instance) -> list[RawCatalogItem]:
    instance_gpu_brand = instance["brand_short"]
    dstack_gpu_name = GPU_MAP.get(instance_gpu_brand)
    if dstack_gpu_name is None:
        logger.warning(f"Failed to find GPU name matching '{instance_gpu_brand}'")
        return []

    instance_types = []
    for variant in instance["variants"]:
        for location, _count in variant["available_nodes_per_dc"].items():
            raw = RawCatalogItem(
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
            )
            instance_types.append(raw)

    return instance_types


GPU_MAP = {
    r"RTX 4090": "RTX4090",
    r"RTX 5090": "RTX5090",
    r"RTX 6000 Pro": "RTX6000PRO",
}


def _make_request(endpoint: str, request_data: dict) -> Union[dict, str, None]:
    response = requests.request(
        "POST",
        f"{CLOUDRIFT_SERVER_ADDRESS}/api/v1/{endpoint}",
        json={"version": CLOUDRIFT_API_VERSION, "data": request_data},
        timeout=5.0,
    )
    if not response.ok:
        response.raise_for_status()
    try:
        response_json = response.json()
        if isinstance(response_json, str):
            return response_json
        return response_json["data"]
    except requests.exceptions.JSONDecodeError:
        return None
