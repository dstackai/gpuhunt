import logging
from typing import Any, Optional

import requests
from requests import Response

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt._internal.constraints import KNOWN_AMD_GPUS, KNOWN_NVIDIA_GPUS
from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

API_URL = "https://api.vultr.com/v2"

EXCLUSION_LIST = ["GH200"]


class VultrProvider(AbstractProvider):
    NAME = "vultr"

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        offers = fetch_offers()
        return sorted(offers, key=lambda i: i.price)


def fetch_offers() -> Optional[list[RawCatalogItem]]:
    """Fetch plans with types:
    1. Cloud GPU (vcg),
    2. Bare Metal (vbm),
    3. and other CPU plans, including:
        Cloud Compute (vc2),
        High Frequency Compute (vhf),
        High Performance (vhp),
        All optimized Cloud Types (voc)"""
    bare_metal_plans_response = _make_request("GET", "/plans-metal?per_page=500")
    other_plans_response = _make_request("GET", "/plans?type=all&per_page=500")
    return convert_response_to_raw_catalog_items(bare_metal_plans_response, other_plans_response)


def convert_response_to_raw_catalog_items(
    bare_metal_plans_response: Response, other_plans_response: Response
) -> list[RawCatalogItem]:
    catalog_items = []

    bare_metal_plans = bare_metal_plans_response.json()["plans_metal"]
    other_plans = other_plans_response.json()["plans"]

    for plan in bare_metal_plans:
        for location in plan["locations"]:
            catalog_item = get_bare_metal_plans(plan, location)
            if catalog_item:
                catalog_items.append(catalog_item)

    for plan in other_plans:
        for location in plan["locations"]:
            catalog_item = get_instance_plans(plan, location)
            if catalog_item:
                catalog_items.append(catalog_item)

    return catalog_items


def get_bare_metal_plans(plan: dict, location: str) -> Optional[RawCatalogItem]:
    gpu_name, gpu_count = extract_gpu_info_from_id(plan["id"])
    if gpu_name in EXCLUSION_LIST:
        logger.info(f"Excluding plan with GPU {gpu_name} as it is not supported.")
        return None
    gpu_memory = get_gpu_memory(gpu_name) if gpu_name else None
    gpu_vendor = get_gpu_vendor(gpu_name)
    return RawCatalogItem(
        instance_name=plan["id"],
        location=location,
        price=round(plan["monthly_cost"] / 730, 2),
        cpu=plan["cpu_count"],
        memory=plan["ram"] / 1024,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        gpu_vendor=gpu_vendor,
        spot=False,
        disk_size=plan["disk"],
    )


def get_instance_plans(plan: dict, location: str) -> Optional[RawCatalogItem]:
    plan_type = plan["type"]
    if plan_type in ["vc2", "vhf", "vhp", "voc"]:
        return RawCatalogItem(
            instance_name=plan["id"],
            location=location,
            price=plan["hourly_cost"],
            cpu=plan["vcpu_count"],
            memory=plan["ram"] / 1024,
            gpu_count=0,
            gpu_name=None,
            gpu_memory=None,
            gpu_vendor=None,
            spot=False,
            disk_size=plan["disk"],
        )
    elif plan_type == "vcg":
        gpu_name = plan["gpu_type"].split("_")[1] if "_" in plan["gpu_type"] else None
        if gpu_name in EXCLUSION_LIST:
            logger.info(f"Excluding plan with GPU {gpu_name} as it is not supported.")
            return None
        gpu_vendor = get_gpu_vendor(gpu_name)
        gpu_memory_gb = plan["gpu_vram_gb"]
        gpu_count = (
            max(1, gpu_memory_gb // get_gpu_memory(gpu_name, gpu_memory_gb)) if gpu_name else 0
        )  # For fractional GPU,
        # gpu_count=1
        return RawCatalogItem(
            instance_name=plan["id"],
            location=location,
            price=plan["hourly_cost"],
            cpu=plan["vcpu_count"],
            memory=plan["ram"] / 1024,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_memory=gpu_memory_gb,
            gpu_vendor=gpu_vendor,
            spot=False,
            disk_size=plan["disk"],
        )


# def get_gpu_memory(gpu_name: str) -> float:
#     for gpu in KNOWN_NVIDIA_GPUS:
#         if gpu.name.upper() == gpu_name:
#             return gpu.memory
#     for gpu in KNOWN_AMD_GPUS:
#         if gpu.name.upper() == gpu_name:
#             return gpu.memory
#     logger.error(f"GPU {gpu_name} not found in known GPU lists.")
#     raise ValueError(f"GPU {gpu_name} not found.")
def get_gpu_memory(gpu_name: str, memory: Optional[int] = None) -> float:
    if memory:
        for gpu in KNOWN_NVIDIA_GPUS:
            if gpu.name == gpu_name.upper() and gpu.memory == memory:
                return gpu.memory

    for gpu in KNOWN_NVIDIA_GPUS:
        if gpu.name == gpu_name.upper():
            return gpu.memory

    for gpu in KNOWN_AMD_GPUS:
        if gpu.name == gpu_name.upper() and (memory is None or gpu.memory == memory):
            return gpu.memory
    logger.error(f"GPU {gpu_name} with memory {memory} not found in known GPU lists.")
    raise ValueError(f"GPU {gpu_name} with memory {memory} not found.")


def get_gpu_vendor(gpu_name: Optional[str]) -> Optional[str]:
    if gpu_name is None:
        return None
    for gpu in KNOWN_NVIDIA_GPUS:
        if gpu.name == gpu_name.upper():
            return AcceleratorVendor.NVIDIA.value
    for gpu in KNOWN_AMD_GPUS:
        if gpu.name == gpu_name.upper():
            return AcceleratorVendor.AMD.value
    return None


def extract_gpu_info_from_id(id_str: str):
    parts = id_str.split("-")
    if "gpu" in parts:
        gpu_name = parts[-2].upper()
        try:
            gpu_count = int(parts[-3])
        except ValueError:
            gpu_count = 1  # Default set to 1 if count is not explicitly specified,
            # for instance in vbm-64c-2048gb-l40-gpu count is not specified but
            # in vbm-64c-2048gb-8-l40-gpu count is specified as 8
        return gpu_name, gpu_count
    return None, 0


def _make_request(method: str, path: str, data: Any = None) -> Response:
    response = requests.request(
        method=method,
        url=API_URL + path,
        json=data,
        timeout=30,
    )
    response.raise_for_status()
    return response


# todo delete
vultr = VultrProvider()
# print(len(vultr.get()))
# print(vultr.get())
print(get_gpu_memory("A100", 80))
