import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from requests import Response

from gpuhunt import CatalogItem, QueryFilter
from gpuhunt._internal.constraints import (
    find_accelerators,
    get_gpu_vendor,
    is_nvidia_superchip,
)
from gpuhunt._internal.models import AcceleratorVendor, CPUArchitecture
from gpuhunt.providers.base import OnlineProvider

logger = logging.getLogger(__name__)

API_URL = "https://api.vultr.com/v2"


class VultrProvider(OnlineProvider):
    NAME = "vultr"

    def __init__(self, api_key: str | None = None, regions: list[str] | None = None):
        self.api_key = api_key
        self.regions = regions

    @classmethod
    def from_env(cls) -> "VultrProvider":
        return cls(api_key=os.getenv("VULTR_API_KEY"))

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        offers = fetch_offers(api_key=self.api_key, regions=self.regions)
        return sorted(offers, key=lambda i: i.price)


def fetch_offers(
    api_key: str | None = None, regions: list[str] | None = None
) -> list[CatalogItem]:
    """Fetch plans with types:
    1. GPU plans (vcg, vdm),
    2. Bare Metal (vbm),
    3. and other CPU plans, including:
        Cloud Compute (vc2),
        High Frequency Compute (vhf),
        High Performance (vhp),
        All optimized Cloud Types (voc)"""
    bare_metal_plans_response = _make_request("GET", "/plans-metal?per_page=500")
    other_plans_response = _make_request("GET", "/plans?type=all&per_page=500")
    vdm_locations = None
    if api_key:
        try:
            vdm_locations = _get_vdm_locations(api_key, regions)
        except requests.RequestException as e:
            logger.warning("Failed to fetch Vultr VDM availability: %s", e)
            # Preserve existing offers without using unverified public VDM locations.
            vdm_locations = {}
    return _make_offers(bare_metal_plans_response, other_plans_response, vdm_locations)


def _get_vdm_locations(api_key: str, regions: list[str] | None) -> dict[str, list[str]]:
    if not regions:
        response = _make_request("GET", "/regions?per_page=500", api_key=api_key)
        regions = [region["id"] for region in response.json()["regions"]]

    locations: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            region: executor.submit(
                _make_request, "GET", f"/regions/{region}/availability?type=vdm", api_key=api_key
            )
            for region in regions
        }
        for region, future in futures.items():
            for plan_id in future.result().json()["available_plans"]:
                locations.setdefault(plan_id, []).append(region)
    return locations


def _make_offers(
    bare_metal_plans_response: Response,
    other_plans_response: Response,
    vdm_locations: dict[str, list[str]] | None = None,
) -> list[CatalogItem]:
    offers: list[CatalogItem] = []

    bare_metal_plans = bare_metal_plans_response.json()["plans_metal"]
    other_plans = other_plans_response.json()["plans"]

    for plans, make_offer in (
        (bare_metal_plans, get_bare_metal_plans),
        (other_plans, get_instance_plans),
    ):
        for plan in plans:
            # Only publish on-demand offers. Accept responses that omit this flag.
            if plan.get("deploy_ondemand") is False:
                continue
            if plan.get("type") == "vdm" and vdm_locations is not None:
                # Public VDM locations can be empty despite availability for this account.
                plan["locations"] = vdm_locations.get(plan["id"], [])
            for location in _iter_locations(plan):
                catalog_item = make_offer(plan, location)
                if catalog_item:
                    offers.append(catalog_item)

    # Vultr's free tier plan is priced at 0, which would rank it above every paid plan.
    # Offers are expected to carry a real price, so zero-priced plans are not published.
    return [offer for offer in offers if offer.price > 0]


def _iter_locations(plan: dict) -> Iterator[str]:
    # The plans API sometimes lists an empty string among a plan's locations
    for location in plan["locations"]:
        if not location:
            logger.warning("Skipping empty location of plan %s", plan["id"])
            continue
        yield location


def get_bare_metal_plans(plan: dict, location: str) -> CatalogItem | None:
    cpu_arch = CPUArchitecture.X86
    gpu_count, gpu_name, gpu_memory, gpu_vendor = 0, None, None, None
    if "gpu" in plan["id"]:
        if plan["id"] not in BARE_METAL_GPU_DETAILS:
            logger.warning("Skipping unknown GPU plan %s", plan["id"])
            return None
        gpu_count, gpu_name, gpu_memory = BARE_METAL_GPU_DETAILS[plan["id"]]
        if is_nvidia_superchip(gpu_name):
            cpu_arch = CPUArchitecture.ARM
        gpu_vendor = get_gpu_vendor(gpu_name)
        if gpu_vendor is None:
            logger.warning("Unknown GPU vendor for plan %s, skipping", plan["id"])
            return None
    return CatalogItem(
        provider=VultrProvider.NAME,
        instance_name=plan["id"],
        location=location,
        price=plan["hourly_cost"],
        cpu_arch=cpu_arch,
        cpu=plan["cpu_threads"],
        memory=plan["ram"] / 1024,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        gpu_vendor=gpu_vendor,
        spot=False,
        disk_size=plan["disk"],
    )


def get_instance_plans(plan: dict, location: str) -> CatalogItem | None:
    cpu_arch = CPUArchitecture.X86
    plan_type = plan["type"]
    if plan_type in ["vc2", "vhf", "vhp", "voc"]:
        return CatalogItem(
            provider=VultrProvider.NAME,
            instance_name=plan["id"],
            location=location,
            price=plan["hourly_cost"],
            cpu_arch=cpu_arch,
            cpu=plan["vcpu_count"],
            memory=plan["ram"] / 1024,
            gpu_count=0,
            gpu_name=None,
            gpu_memory=None,
            gpu_vendor=None,
            spot=False,
            disk_size=plan["disk"],
        )
    elif plan_type in ["vcg", "vdm"]:
        gpu_details = _get_instance_gpu_details(plan)
        if gpu_details is None:
            return None
        gpu_count, gpu_name, gpu_memory = gpu_details
        gpu_vendor = get_gpu_vendor(gpu_name)
        if not gpu_vendor:
            logger.warning(
                "Failed to detect GPU vendor %s for plan %s, skipping", gpu_name, plan["id"]
            )
            return None
        if is_nvidia_superchip(gpu_name):
            cpu_arch = CPUArchitecture.ARM
        return CatalogItem(
            provider=VultrProvider.NAME,
            instance_name=plan["id"],
            location=location,
            price=plan["hourly_cost"],
            cpu_arch=cpu_arch,
            cpu=plan["vcpu_count"],
            memory=plan["ram"] / 1024,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_memory=gpu_memory,
            gpu_vendor=gpu_vendor,
            spot=False,
            disk_size=plan["disk"],
        )
    return None


def _get_instance_gpu_details(plan: dict) -> tuple[int, str, float] | None:
    """Return GPU count, model name, and memory per GPU in GB."""
    if plan["type"] == "vdm":
        # VDM plans omit GPU metadata, so use known configurations as for bare metal.
        gpu_details = VDM_GPU_DETAILS.get(plan["id"])
        if gpu_details is None:
            logger.warning("Skipping unknown VDM GPU plan %s", plan["id"])
        return gpu_details

    gpu_type = plan.get("gpu_type")
    if not gpu_type or "_" not in gpu_type:
        logger.warning(
            "Missing or invalid gpu_type %s for plan %s, skipping", gpu_type, plan["id"]
        )
        return None
    gpu_name = gpu_type.split("_")[1]
    full_gpu_memory = get_gpu_memory(gpu_name)
    if not full_gpu_memory:
        logger.warning(
            "Failed to detect GPU memory %s for plan %s, skipping", gpu_type, plan["id"]
        )
        return None
    total_gpu_memory = plan["gpu_vram_gb"]
    # For fractional GPU, gpu_count=1
    gpu_count = max(1, total_gpu_memory // full_gpu_memory)
    return gpu_count, gpu_name, total_gpu_memory / gpu_count


def get_gpu_memory(gpu_name: str) -> int | None:
    if gpu_name.upper() == "A100":
        return 80  # VULTR A100 instances have 80GB
    if accelerators := find_accelerators(
        names=[gpu_name], vendors=[AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD]
    ):
        return accelerators[0].memory
    logger.warning(f"Unknown GPU {gpu_name}")
    return None


def _make_request(
    method: str, path: str, data: Any = None, *, api_key: str | None = None
) -> Response:
    response = requests.request(
        method=method,
        url=API_URL + path,
        json=data,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        timeout=30,
    )
    response.raise_for_status()
    return response


BARE_METAL_GPU_DETAILS = {
    "vbm-48c-1024gb-4-a100-gpu": (4, "A100", 80),
    "vbm-112c-2048gb-8-h100-gpu": (8, "H100", 80),
    "vbm-112c-2048gb-8-a100-gpu": (8, "A100", 80),
    "vbm-64c-2048gb-8-l40-gpu": (8, "L40S", 48),
    "vbm-72c-480gb-gh200-gpu": (1, "GH200", 96),
    "vbm-256c-2048gb-8-mi300x-gpu": (8, "MI300X", 192),
    "vbm-256c-3072gb-8-mi325x-gpu": (8, "MI325X", 256),
    "vbm-256c-3072gb-8-b200-gpu": (8, "B200", 192),
    "vbm-256c-3072gb-8-mi355x-gpu": (8, "MI355X", 288),
}

VDM_GPU_DETAILS = {
    "vcg-a16-6c-64g-16vram": (1, "A16", 16),
    "vcg-a16-96c-960g-256vram": (16, "A16", 16),
    "vcg-a16-96c-878g-256vram": (16, "A16", 16),
    "vcg-a40-24c-120g-48vram": (1, "A40", 48),
    "vcg-a40-96c-480g-192vram": (4, "A40", 48),
    "vcg-a100-12c-120g-80vram": (1, "A100", 80),
    "vcg-a100-96c-960g-640vram": (8, "A100", 80),
    "vcg-h100-216c-1914gb-640vram": (8, "H100", 80),
    "vcg-b200-248c-2826g-1536vram": (8, "B200", 192),
    "vcg-mi355x-252c-2872g-2304vram": (8, "MI355X", 288),
    # Use Vultr's documented 8 x 256 GB MI325X configuration.
    # The VDM plan's 1536vram suffix conflicts; its allocation is not verified.
    "vcg-mi325x-252c-2872g-1536vram": (8, "MI325X", 256),
}
