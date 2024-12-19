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
        return sorted(offers, key=lambda i: i.price) if offers is not None else []


def fetch_offers() -> Optional[list[RawCatalogItem]]:
    """Fetch plans with types:
    1. Cloud GPU (vcg),
    2. Bare Metal (vbm),
    3. and others cpu plans, which includes:
                             Cloud Compute (vc2),
                             High Frequency Compute (vhf),
                             High Performance (vhp),
                             All optimized Cloud Types (voc)"""
    try:
        cloud_gpu_plans_response = _make_request("GET", "/plans?type=vcg")
        bare_metal_plans_response = _make_request("GET", "/plans-metal")
        other_plans_response = _make_request("GET", "/plans?type=all")
        combined_response = {
            "plans": (
                cloud_gpu_plans_response.json().get("plans", [])
                + other_plans_response.json().get("plans", [])
            ),
            "plans_metal": bare_metal_plans_response.json().get("plans_metal", []),
        }
        return convert_response_to_raw_catalog_items(combined_response)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch plans: {str(e)}")
        return None


def convert_response_to_raw_catalog_items(response: dict) -> list[RawCatalogItem]:
    catalog_items = []
    plans = response.get("plans", []) + response.get("plans_metal", [])

    for plan in plans:
        for location in plan.get("locations", []):
            if plan in response.get("plans_metal", []):
                catalog_item = get_bare_metal_plans(plan, location)
            else:
                catalog_item = get_instance_plans(plan, location)
            if catalog_item:
                catalog_items.append(catalog_item)

    return catalog_items


def get_bare_metal_plans(plan: dict, location: str) -> Optional[RawCatalogItem]:
    gpu_name, gpu_count = extract_gpu_info_from_id(plan.get("id", ""))
    if gpu_name in EXCLUSION_LIST:
        logger.info(f"Excluding plan with GPU {gpu_name} as it is not supported.")
        return None
    gpu_memory = get_gpu_memory(gpu_name) * gpu_count if gpu_name else None
    gpu_vendor = get_gpu_vendor(gpu_name)
    return RawCatalogItem(
        instance_name=plan.get("id"),
        location=location,
        price=round(plan.get("monthly_cost", 0) / 730, 2),
        cpu=plan.get("cpu_count"),
        memory=plan.get("ram", 0) / 1024,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        gpu_vendor=gpu_vendor,
        spot=False,
        disk_size=plan.get("disk", 0),
    )


def get_instance_plans(plan: dict, location: str) -> Optional[RawCatalogItem]:
    plan_type = plan.get("type", "")
    if plan_type in ["vc2", "vhf", "vhp", "voc"]:
        return RawCatalogItem(
            instance_name=plan.get("id"),
            location=location,
            price=plan.get("hourly_cost", 0),
            cpu=plan.get("vcpu_count"),
            memory=plan.get("ram", 0) / 1024,
            gpu_count=0,
            gpu_name=None,
            gpu_memory=None,
            gpu_vendor=None,
            spot=False,
            disk_size=plan.get("disk", 0),
        )
    elif plan_type == "vcg":
        gpu_name = (
            plan.get("gpu_type", "").split("_")[1] if "_" in plan.get("gpu_type", "") else None
        )
        if gpu_name in EXCLUSION_LIST:
            logger.info(f"Excluding plan with GPU {gpu_name} as it is not supported.")
            return None
        gpu_vendor = get_gpu_vendor(gpu_name)
        gpu_memory_gb = plan.get("gpu_vram_gb", 0)
        gpu_count = (
            max(1, gpu_memory_gb // get_gpu_memory(gpu_name)) if gpu_name else 0
        )  # For fractional GPU,
        # gpu_count=1
        return RawCatalogItem(
            instance_name=plan.get("id"),
            location=location,
            price=plan.get("hourly_cost", 0),
            cpu=plan.get("vcpu_count"),
            memory=plan.get("ram", 0) / 1024,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_memory=gpu_memory_gb,
            gpu_vendor=gpu_vendor,
            spot=False,
            disk_size=plan.get("disk", 0),
        )


def get_gpu_memory(gpu_name: str) -> float:
    for gpu in KNOWN_NVIDIA_GPUS:
        if gpu.name.upper() == gpu_name:
            return gpu.memory
    for gpu in KNOWN_AMD_GPUS:
        if gpu.name.upper() == gpu_name:
            return gpu.memory
    logger.error(f"GPU {gpu_name} not found in known GPU lists.")
    raise ValueError(f"GPU {gpu_name} not found.")


def get_gpu_vendor(gpu_name: Optional[str]) -> Optional[str]:
    if gpu_name is None:
        return None
    for gpu in KNOWN_NVIDIA_GPUS:
        if gpu.name.upper() == gpu_name:
            return AcceleratorVendor.NVIDIA.value
    for gpu in KNOWN_AMD_GPUS:
        if gpu.name.upper() == gpu_name:
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
    try:
        response = requests.request(
            method=method,
            url=API_URL + path,
            json=data,
            timeout=30,
        )
        response.raise_for_status()
        return response

    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        if status_code == requests.codes.not_found:
            logger.exception(f"Resource not found at {API_URL + path}.")
        elif status_code == requests.codes.bad_request:
            logger.exception(
                f"Bad request to {API_URL + path}. Check the request payload or parameters."
            )
        elif status_code == requests.codes.forbidden:
            logger.exception(f"Access forbidden to {API_URL + path}. Check API permissions.")
        elif status_code == requests.codes.unauthorized:
            logger.exception(
                f"Unauthorized access to {API_URL + path}. Check API key or authentication details."
            )
        else:
            logger.exception(f"HTTP error {status_code} occurred when accessing {API_URL + path}.")

        raise
    except requests.RequestException as e:
        logger.exception(f"Request error while accessing {API_URL + path}: {str(e)}")
        raise
