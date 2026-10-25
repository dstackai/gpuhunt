import logging
from typing import cast

import requests
from typing_extensions import TypedDict

from gpuhunt._internal.constraints import is_nvidia_superchip
from gpuhunt._internal.models import (
    AcceleratorVendor,
    CatalogItem,
    CPUArchitecture,
    JSONObject,
    QueryFilter,
)
from gpuhunt.providers.base import OnlineProvider, get_creds_env

logger = logging.getLogger(__name__)

# Documented at https://www.runcrate.ai/docs/api-reference/instances/list-available-instance-types.
# Returns every offer that can be launched right now in one unpaginated response.
INSTANCE_TYPES_URL = "https://www.runcrate.ai/api/v1/instances/types"
TIMEOUT = 30

# Runcrate `gpu_type` to the gpuhunt GPU name and per-GPU memory in GiB. The memory sizes are
# the ones in `KNOWN_NVIDIA_GPUS`. A type that is not listed here is skipped with a warning.
GPU_MAP: dict[str, tuple[str, float]] = {
    "A10": ("A10", 24),
    "A100": ("A100", 40),
    "A100_80G": ("A100", 80),
    "A4000": ("A4000", 16),
    "A40": ("A40", 48),
    "A5000": ("A5000", 24),
    "A6000": ("A6000", 48),
    "B200": ("B200", 180),
    "B300": ("B300", 270),
    "GH200": ("GH200", 96),
    "H100": ("H100", 80),
    "H200": ("H200", 141),
    "L4": ("L4", 24),
    "L40": ("L40", 48),
    "L40S": ("L40S", 48),
    "RTX4000Ada": ("RTX4000Ada", 20),
    "RTX6000Ada": ("RTX6000Ada", 48),
    "RTXPro6000": ("RTXPRO6000", 96),
    "V100": ("V100", 16),
    "V100_32G": ("V100", 32),
}


class RuncrateProvider(OnlineProvider):
    """Online provider for Runcrate (https://www.runcrate.ai) GPU and CPU VMs.

    The instance types endpoint needs an API key, so offers are queried live with
    `RUNCRATE_API_KEY`. Every offer is a fixed VM shape in one region at an on-demand hourly
    price for the whole VM. `instance_name` is the offer `id`, which the create instance API
    takes as `instance_type_id`. Offer ids change as capacity changes, so they are only valid
    for a launch shortly after the query.
    """

    NAME = "runcrate"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "RuncrateProvider":
        return cls(api_key=get_creds_env("RUNCRATE_API_KEY"))

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        response = requests.get(
            INSTANCE_TYPES_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("Unexpected response from Runcrate instance types endpoint")

        offers: list[CatalogItem] = []
        for instance_type in data["data"]:
            if not isinstance(instance_type, dict):
                logger.warning("Skipping malformed Runcrate instance type: %r", instance_type)
                continue
            offer = _make_offer(instance_type)
            if offer is not None:
                offers.append(offer)
        return sorted(offers, key=lambda i: i.price)


class RuncrateCatalogItemProviderData(TypedDict):
    # Runcrate GPU type, which the create instance API also accepts with `gpu_count` and
    # `region` in place of `instance_type_id`.
    gpu_type: str


def _make_offer(instance_type: dict) -> CatalogItem | None:
    offer_id = instance_type.get("id")
    if not isinstance(offer_id, str) or not offer_id:
        logger.warning("Skipping Runcrate instance type without id: %r", instance_type)
        return None
    try:
        gpu_type = _required_str(instance_type, "gpu_type")
        region = _required_str(instance_type, "region")
        gpu_count = _required_int(instance_type, "gpu_count")
        cpu = _required_int(instance_type, "cpu_cores")
        memory = _required_number(instance_type, "memory_gb")
        disk_size = _required_number(instance_type, "storage_gb")
        price = _required_number(instance_type, "hourly_rate")
    except ValueError as e:
        logger.warning("Skipping Runcrate instance type %s: %s", offer_id, e)
        return None
    if gpu_count < 0 or cpu <= 0 or memory <= 0 or disk_size <= 0 or price <= 0:
        logger.warning(
            "Skipping Runcrate instance type %s: non-positive resources or price", offer_id
        )
        return None

    gpu_name = None
    gpu_memory = None
    gpu_vendor = None
    cpu_arch = CPUArchitecture.X86
    if gpu_count > 0:
        if gpu_type not in GPU_MAP:
            logger.warning(
                "Skipping Runcrate instance type %s: unknown gpu_type %r", offer_id, gpu_type
            )
            return None
        gpu_name, gpu_memory = GPU_MAP[gpu_type]
        gpu_vendor = AcceleratorVendor.NVIDIA
        if is_nvidia_superchip(gpu_name):
            cpu_arch = CPUArchitecture.ARM

    provider_data = RuncrateCatalogItemProviderData(gpu_type=gpu_type)
    return CatalogItem(
        provider=RuncrateProvider.NAME,
        instance_name=offer_id,
        location=region,
        price=price,
        cpu=cpu,
        memory=memory,
        gpu_vendor=gpu_vendor,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        spot=False,
        disk_size=disk_size,
        cpu_arch=cpu_arch,
        provider_data=cast(JSONObject, dict(provider_data)),
    )


def _required_str(instance_type: dict, name: str) -> str:
    value = instance_type.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing {name}")
    return value


def _required_int(instance_type: dict, name: str) -> int:
    value = instance_type.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"missing {name}")
    return value


def _required_number(instance_type: dict, name: str) -> float:
    value = instance_type.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"missing {name}")
    return float(value)
