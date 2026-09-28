import logging
import math
import os
import threading
from dataclasses import replace
from typing import Any, cast

import requests
from typing_extensions import NotRequired, TypedDict

from gpuhunt import CatalogItem, QueryFilter
from gpuhunt._internal.constraints import matches
from gpuhunt._internal.models import AcceleratorVendor, JSONObject
from gpuhunt.providers.base import OnlineProvider

logger = logging.getLogger(__name__)

PRICING_URL = "https://billing.app.daytona.io/gpu-pricing"
API_URL = "https://app.daytona.io/api"
TIMEOUT = 10.0
GPU_REGION = "earth"
MAX_GPU_COUNT = 8

# Preferences for balanced offers, not fixed instance configurations. Explicit
# query bounds take precedence. Daytona meters each resource independently.
CPUS_PER_GPU = 8
DISK_PER_GPU = 256
MEMORY_PER_GPU = 100
CONSUMER_MEMORY_PER_GPU = 50

# Published GPU sandbox limits: https://www.daytona.io/docs/en/sandboxes/#gpu-sandboxes
MAX_CPUS_PER_GPU = 16
MAX_MEMORY_PER_GPU = 192
MAX_DISK_PER_GPU = 512

# Only types supported by sandbox creation belong here. B200 has billing rates,
# but the create API rejects gpuType=["B200"] as an invalid GPU type.
GPU_MAP: dict[str, tuple[str, AcceleratorVendor, float, float]] = {
    "B300": ("B300", AcceleratorVendor.NVIDIA, 270.0, MEMORY_PER_GPU),
    "H100": ("H100", AcceleratorVendor.NVIDIA, 80.0, MEMORY_PER_GPU),
    "H200": ("H200", AcceleratorVendor.NVIDIA, 141.0, MEMORY_PER_GPU),
    "MI355X": ("MI355X", AcceleratorVendor.AMD, 288.0, MEMORY_PER_GPU),
    "RTX-PRO-6000": ("RTXPRO6000", AcceleratorVendor.NVIDIA, 96.0, MEMORY_PER_GPU),
    "RTX-5090": ("RTX5090", AcceleratorVendor.NVIDIA, 32.0, CONSUMER_MEMORY_PER_GPU),
    "RTX-4090": ("RTX4090", AcceleratorVendor.NVIDIA, 24.0, CONSUMER_MEMORY_PER_GPU),
}


class DaytonaCatalogItemProviderData(TypedDict):
    # Daytona GPU name, included only when it differs from the gpuhunt GPU name.
    gpu_type: NotRequired[str]


class DaytonaProvider(OnlineProvider):
    NAME = "daytona"

    def __init__(self, api_key: str | None = None, api_url: str = API_URL):
        self.api_key = (api_key or "").strip() or None
        self.api_url = api_url.rstrip("/")
        self._organization_id: str | None = None
        self._warned_no_auth = False
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "DaytonaProvider":
        return cls(
            api_key=os.getenv("DAYTONA_API_KEY"),
            api_url=os.getenv("DAYTONA_API_URL", API_URL),
        )

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        if not self.api_key:
            with self._lock:
                if not self._warned_no_auth:
                    logger.warning(
                        "DAYTONA_API_KEY is not set. Returning price estimates without "
                        "checking live availability."
                    )
                    self._warned_no_auth = True
        query = query_filter or QueryFilter()
        pricing = _fetch_pricing()
        offers = []
        if query.max_gpu_count is None or query.max_gpu_count > 0:
            capacity = self._get_gpu_capacity() if self.api_key else None
            offers.extend(_make_gpu_offers(pricing, query, balance_resources, capacity))
        cpu_offer = _make_offer(pricing, query)
        if cpu_offer is not None:
            offers.extend(
                replace(cpu_offer, location=region)
                for region in _fetch_shared_regions(self.api_url)
            )
        return sorted(offers, key=lambda i: i.price)

    def _get_gpu_capacity(self) -> dict[str, dict[str, int]]:
        with self._lock:
            if self._organization_id is None:
                identity = _request_json(f"{self.api_url}/api-keys/current", self.api_key)
                if not (
                    isinstance(identity, dict)
                    and isinstance(identity.get("organizationId"), str)
                    and identity["organizationId"]
                ):
                    raise ValueError("Unexpected Daytona API key response: missing organizationId")
                self._organization_id = identity["organizationId"]
        data = _request_json(
            f"{self.api_url}/organizations/{self._organization_id}/gpu-capacity", self.api_key
        )
        if not isinstance(data, dict) or not isinstance(data.get("capacity"), list):
            raise ValueError("Unexpected Daytona GPU capacity response")
        capacity = {}
        for row in data["capacity"]:
            if not isinstance(row, dict) or not isinstance(row.get("gpuType"), str):
                raise ValueError("Unexpected Daytona GPU capacity entry")
            capacity[row["gpuType"]] = {
                "onDemand": _count(row.get("availableOnDemand")),
                "spot": _count(row.get("availableSpot")),
            }
        return capacity


def _request_json(url: str, api_key: str | None = None) -> Any:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _fetch_pricing() -> dict[str, Any]:
    pricing = _request_json(PRICING_URL)
    if (
        not isinstance(pricing, dict)
        or not isinstance(pricing.get("gpus"), list)
        or not isinstance(pricing.get("resources"), dict)
        or not isinstance(pricing["resources"].get("onDemand"), dict)
        or not isinstance(pricing["resources"].get("spot"), dict)
    ):
        raise ValueError("Unexpected Daytona gpu-pricing response")
    return pricing


def _fetch_shared_regions(api_url: str) -> list[str]:
    regions = _request_json(f"{api_url}/shared-regions")
    if not isinstance(regions, list) or any(
        not isinstance(region, dict) or not isinstance(region.get("id"), str) or not region["id"]
        for region in regions
    ):
        raise ValueError("Unexpected Daytona shared regions response")
    return [region["id"] for region in regions]


def _count(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
    ):
        raise ValueError("Unexpected Daytona resource count")
    return int(value)


def _resource_size(
    preferred: float, minimum: float | None, maximum: float | None, limit: float = math.inf
) -> int | None:
    lower = math.ceil(max(1, minimum if minimum is not None else 1))
    upper = min(limit, maximum if maximum is not None else limit)
    if lower > upper:
        return None
    return math.floor(min(upper, max(lower, math.ceil(preferred))))


def _make_gpu_offers(
    pricing: dict[str, Any],
    query: QueryFilter,
    balance_resources: bool,
    capacity: dict[str, dict[str, int]] | None,
) -> list[CatalogItem]:
    offers = []
    for gpu in pricing["gpus"]:
        gpu_type = gpu["type"]
        if gpu_type not in GPU_MAP:
            # B200 is a known billing-only entry, not a newly introduced type.
            if gpu_type != "B200":
                logger.warning("Failed to find GPU name matching '%s'", gpu_type)
            continue
        for pricing_type in ("onDemand", "spot"):
            available = MAX_GPU_COUNT
            if capacity is not None:
                available = min(available, capacity.get(gpu_type, {}).get(pricing_type, 0))
            for gpu_count in range(1, available + 1):
                offer = _make_offer(
                    pricing,
                    query,
                    gpu=gpu,
                    gpu_count=gpu_count,
                    pricing_type=pricing_type,
                    balance_resources=balance_resources,
                )
                if offer is not None:
                    offers.append(offer)
    return offers


def _make_offer(
    pricing: dict[str, Any],
    query: QueryFilter,
    *,
    gpu: dict[str, Any] | None = None,
    gpu_count: int = 0,
    pricing_type: str = "onDemand",
    balance_resources: bool = True,
) -> CatalogItem | None:
    if gpu is None:
        # CPU sandboxes have no published provider-wide resource limits.
        gpu_name, gpu_vendor, gpu_memory = None, None, None
        default_cpu, default_memory, default_disk = 1, 1, 3
        cpu_limit = memory_limit = disk_limit = math.inf
        gpu_price = 0
        provider_data = {}
    else:
        gpu_name, gpu_vendor, gpu_memory, memory_per_gpu = GPU_MAP[gpu["type"]]
        default_cpu = CPUS_PER_GPU * gpu_count if balance_resources else 1
        default_memory = memory_per_gpu * gpu_count if balance_resources else 1
        default_disk = DISK_PER_GPU * gpu_count if balance_resources else 1
        cpu_limit = MAX_CPUS_PER_GPU * gpu_count
        memory_limit = MAX_MEMORY_PER_GPU * gpu_count
        disk_limit = MAX_DISK_PER_GPU * gpu_count
        gpu_price = gpu_count * gpu[f"{pricing_type}PricePerHour"]
        provider_data = _gpu_provider_data(gpu["type"], gpu_name)

    cpu = _resource_size(default_cpu, query.min_cpu, query.max_cpu, cpu_limit)
    memory = _resource_size(default_memory, query.min_memory, query.max_memory, memory_limit)
    disk_size = _resource_size(default_disk, query.min_disk_size, query.max_disk_size, disk_limit)
    if cpu is None or memory is None or disk_size is None:
        return None
    rates = pricing["resources"][pricing_type]
    instance_name = f"{cpu}cpu-{memory}gb-{disk_size}gb"
    if gpu is not None:
        instance_name = f"{gpu_count}x-{gpu_name}-{instance_name}"
    offer = CatalogItem(
        provider=DaytonaProvider.NAME,
        instance_name=instance_name,
        # CPU offers are copied to each shared region after filtering the shape.
        location=GPU_REGION if gpu is not None else "",
        price=round(
            gpu_price
            + cpu * rates["vcpuPerHour"]
            + memory * rates["memoryGiBPerHour"]
            + disk_size * rates["diskGiBPerHour"],
            6,
        ),
        cpu=cpu,
        memory=float(memory),
        disk_size=float(disk_size),
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        gpu_vendor=gpu_vendor,
        spot=pricing_type == "spot",
        provider_data=provider_data,
    )
    return offer if matches(offer, query) else None


def _gpu_provider_data(gpu_type: str, gpu_name: str) -> JSONObject:
    if gpu_type == gpu_name:
        return {}
    return cast(JSONObject, DaytonaCatalogItemProviderData(gpu_type=gpu_type))
