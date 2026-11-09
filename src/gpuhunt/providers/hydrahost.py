import logging
import math
import re
from collections.abc import Iterator
from typing import cast

import requests
from typing_extensions import NotRequired, TypedDict

from gpuhunt._internal.constraints import find_accelerators, is_nvidia_superchip
from gpuhunt._internal.models import (
    AcceleratorVendor,
    CatalogItem,
    CPUArchitecture,
    JSONObject,
    QueryFilter,
)
from gpuhunt.providers.base import OnlineProvider

logger = logging.getLogger(__name__)

# Brokkr marketplace inventory. Listing it needs no credentials; provisioning a listing
# (`PATCH /api/v1/inventory/{id}/provision`) needs an API key and a billing method.
# OpenAPI spec: https://brokkr.hydrahost.com/api/v1/swagger-json
# Docs: https://brokkr.hydrahost.com/docs/brokkr-api-inventory
INVENTORY_URL = "https://brokkr.hydrahost.com/api/v1/inventory"
PAGE_SIZE = 100  # the API maximum
MAX_PAGES = 50
TIMEOUT = 30
# Only "on demand" listings can be claimed right away. "reserve" listings need a reservation
# from Hydra and "preorder" listings are not delivered yet, so neither is a rentable offer.
ON_DEMAND_FILTER = "status:eq:on demand"
INTERRUPTIBLE_FILTER = "status:eq:on demand|interruptibleReady:eq:true"

# `specs.gpu.model` is the nvidia-smi product name. Explicit mapping to `KNOWN_NVIDIA_GPUS`
# names and per-GPU memory in GiB; models that are not listed are skipped with a warning.
GPU_MAP: dict[str, tuple[str, int]] = {
    "nvidia a10": ("A10", 24),
    "nvidia a100 80gb pcie": ("A100", 80),
    "nvidia a100-pcie-40gb": ("A100", 40),
    "nvidia a100-sxm4-40gb": ("A100", 40),
    "nvidia a100-sxm4-80gb": ("A100", 80),
    "nvidia b200": ("B200", 180),
    "nvidia b300 sxm6 ac": ("B300", 270),
    "nvidia b300 sxm6 pc": ("B300", 270),
    "nvidia geforce rtx 3090": ("RTX3090", 24),
    "nvidia geforce rtx 4090": ("RTX4090", 24),
    "nvidia geforce rtx 5090": ("RTX5090", 32),
    "nvidia gh200 480gb": ("GH200", 96),
    "nvidia h100 80gb hbm3": ("H100", 80),
    "nvidia h100 nvl": ("H100NVL", 94),
    "nvidia h100 pcie": ("H100", 80),
    "nvidia h200": ("H200", 141),
    "nvidia h200 nvl": ("H200NVL", 141),
    "nvidia l4": ("L4", 24),
    "nvidia l40": ("L40", 48),
    "nvidia l40s": ("L40S", 48),
    "nvidia rtx 6000 ada generation": ("RTX6000Ada", 48),
    "nvidia rtx a4000": ("A4000", 16),
    "nvidia rtx a5000": ("A5000", 24),
    "nvidia rtx a6000": ("A6000", 48),
    "nvidia rtx pro 6000 blackwell server edition": ("RTXPRO6000", 96),
    "quadro rtx 8000": ("RTX8000", 48),
    "tesla v100-sxm2-16gb": ("V100", 16),
    "tesla v100-sxm2-32gb": ("V100", 32),
}


class HydraHostCatalogItemProviderData(TypedDict):
    # Hostname of the server, as shown in the Brokkr marketplace.
    device_name: str
    # OS slugs accepted by the provision API's `operatingSystem` for this server.
    operating_systems: list[str]
    # "Public" (the server gets a public IPv4) or "NAT".
    network_type: NotRequired[str]
    # Whether the server can run Hydra's confidential computing (TEE) images.
    tee_capable: bool
    # Notice in milliseconds before an interruptible (spot) deployment is reclaimed.
    interruptible_notice_period_ms: NotRequired[float]


class HydraHostProvider(OnlineProvider):
    """Online provider for Hydra Host (https://hydrahost.com) bare metal servers.

    Hydra Host's Brokkr marketplace publishes its inventory without authentication. Every
    "on demand" listing is a whole bare metal server, so each one becomes a single offer with
    all of its GPUs, CPUs and RAM. Servers that accept interruptible contracts also get a spot
    offer at the interruptible price. `instance_name` is the device id that the provision API
    takes, so a dstack backend could claim a server straight from the offer.
    """

    NAME = "hydrahost"

    @classmethod
    def from_env(cls) -> "HydraHostProvider":
        return cls()

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        devices = list(_iter_inventory(ON_DEMAND_FILTER))
        try:
            interruptible_ids = {
                d.get("id") for d in _iter_inventory(INTERRUPTIBLE_FILTER) if isinstance(d, dict)
            }
        except (requests.RequestException, ValueError):
            logger.exception("Failed to fetch interruptible Hydra Host listings")
            interruptible_ids = set()
        offers: list[CatalogItem] = []
        for device in devices:
            if not isinstance(device, dict):
                logger.warning("Skipping malformed Hydra Host listing: %r", device)
                continue
            offers.extend(
                _make_offers(device, interruptible=device.get("id") in interruptible_ids)
            )
        return sorted(offers, key=lambda i: i.price)


def _iter_inventory(filters: str) -> Iterator[dict]:
    page = 1
    while page <= MAX_PAGES:
        response = requests.get(
            INVENTORY_URL,
            params={"page": page, "pageSize": PAGE_SIZE, "filters": filters},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("Unexpected response from Hydra Host inventory API")
        yield from data["data"]
        total_pages = (data.get("meta") or {}).get("totalPages") or 1
        if page >= total_pages or not data["data"]:
            return
        page += 1
    logger.warning("Hydra Host inventory has more than %d pages, stopping", MAX_PAGES)


def _make_offers(device: dict, interruptible: bool) -> list[CatalogItem]:
    device_id = device.get("id")
    if not device_id:
        logger.warning("Skipping Hydra Host listing without id: %r", device)
        return []
    device_id = str(device_id)
    listing = device.get("listing") or {}
    if not listing.get("isActive", True):
        return []

    specs = device.get("specs") or {}
    cpu = get_cpu_count(specs.get("cpu") or {})
    memory = (specs.get("memory") or {}).get("total")
    if not cpu or not isinstance(memory, int | float) or memory <= 0:
        logger.warning("Skipping Hydra Host listing %s: missing CPU or memory", device_id)
        return []
    location = get_location(device.get("location"))
    if not location:
        logger.warning("Skipping Hydra Host listing %s: no location", device_id)
        return []
    disk_size = get_root_disk_gib(device)
    if disk_size is None:
        # with disk_size=None an offer would pass every disk filter
        logger.warning("Skipping Hydra Host listing %s: root disk size unknown", device_id)
        return []

    gpu = specs.get("gpu") or {}
    gpu_count = gpu.get("count") or 0
    gpu_model = gpu.get("model")
    gpu_name: str | None = None
    gpu_memory: float | None = None
    gpu_vendor: AcceleratorVendor | None = None
    if gpu_count:
        known = GPU_MAP.get(str(gpu_model).strip().lower())
        if known is None or not find_accelerators(
            names=[known[0]], vendors=[AcceleratorVendor.NVIDIA]
        ):
            logger.warning("Skipping Hydra Host listing %s: unknown GPU %r", device_id, gpu_model)
            return []
        gpu_name, gpu_memory = known[0], float(known[1])
        gpu_vendor = AcceleratorVendor.NVIDIA
    cpu_arch = get_cpu_arch((specs.get("cpu") or {}).get("model"), gpu_name)

    provider_data = HydraHostCatalogItemProviderData(
        device_name=str(device.get("name") or ""),
        operating_systems=[
            str(os["slug"])
            for os in device.get("availableOperatingSystems") or []
            if isinstance(os, dict) and os.get("slug")
        ],
        tee_capable=bool(device.get("isTeeCapable")),
    )
    network_type = (device.get("networking") or {}).get("networkType")
    if network_type:
        provider_data["network_type"] = str(network_type)

    def make(price: float, spot: bool, data: HydraHostCatalogItemProviderData) -> CatalogItem:
        return CatalogItem(
            provider=HydraHostProvider.NAME,
            instance_name=device_id,
            location=location,
            price=price,
            cpu=cpu,
            memory=float(memory),
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_memory=gpu_memory,
            gpu_vendor=gpu_vendor,
            spot=spot,
            disk_size=float(disk_size),
            cpu_arch=cpu_arch,
            provider_data=cast(JSONObject, dict(data)),
        )

    offers: list[CatalogItem] = []
    on_demand_price = get_hourly_price(listing.get("onDemandPrice"))
    if on_demand_price is not None and not listing.get("isInterruptibleOnly"):
        offers.append(make(on_demand_price, spot=False, data=provider_data))
    interruptible_price = get_hourly_price(listing.get("interruptiblePrice"))
    if interruptible and interruptible_price is not None:
        spot_data = HydraHostCatalogItemProviderData(**provider_data)
        notice = device.get("interruptibleNoticePeriod")
        if isinstance(notice, int | float) and not isinstance(notice, bool):
            spot_data["interruptible_notice_period_ms"] = float(notice)
        offers.append(make(interruptible_price, spot=True, data=spot_data))
    return offers


def get_hourly_price(price: object) -> float | None:
    """
    Whole-server price in dollars per hour from a Brokkr price object, which is in cents.

    >>> get_hourly_price({"perHour": {"perGpu": 59.375, "perCpu": 237.5, "total": 475}})
    4.75
    >>> get_hourly_price({"perHour": {"perGpu": None, "perCpu": None, "total": None}}) is None
    True
    """
    if not isinstance(price, dict):
        return None
    total = (price.get("perHour") or {}).get("total")
    if isinstance(total, bool) or not isinstance(total, int | float) or total <= 0:
        return None
    return round(total / 100, 5)


def get_cpu_count(cpu: dict) -> int:
    """
    Logical CPUs of the server (hardware threads), which is what the OS reports.

    >>> get_cpu_count({"count": 2, "totalCores": 128, "threadsPerCore": 2, "totalThreads": 256})
    256
    >>> get_cpu_count({"count": 2, "totalCores": 44, "threadsPerCore": 2})
    88
    >>> get_cpu_count({})
    0
    """
    threads = cpu.get("totalThreads")
    if isinstance(threads, int) and threads > 0:
        return threads
    cores = cpu.get("totalCores")
    if isinstance(cores, int) and cores > 0:
        return cores * max(1, int(cpu.get("threadsPerCore") or 1))
    return 0


def get_cpu_arch(cpu_model: object, gpu_name: str | None) -> CPUArchitecture:
    """
    >>> get_cpu_arch("AMD EPYC 7763 64-Core Processor", "L40S").value
    'x86'
    >>> get_cpu_arch("Neoverse-V2", None).value
    'arm'
    >>> get_cpu_arch(None, "GH200").value
    'arm'
    """
    if gpu_name is not None and is_nvidia_superchip(gpu_name):
        return CPUArchitecture.ARM
    if cpu_model and re.search(r"neoverse|grace|aarch64|arm", str(cpu_model), re.IGNORECASE):
        return CPUArchitecture.ARM
    return CPUArchitecture.X86


def get_location(location: object) -> str:
    """
    >>> get_location("Singapore")
    'singapore'
    >>> get_location("Kansas City, MO")
    'kansas-city-mo'
    >>> get_location(None)
    ''
    """
    if not location:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", str(location).lower()).strip("-")


def get_root_disk_gib(device: dict) -> int | None:
    """
    Size in GiB of the filesystem mounted at `/` by the default disk layout.

    Servers usually have a small OS disk and large data disks mounted elsewhere (`/data0`),
    so the root filesystem is the space a job can rely on without extra setup. Returns None
    if the layout does not say which disks back `/`.
    """
    root = next(
        (
            layout
            for layout in device.get("defaultDiskLayouts") or []
            if isinstance(layout, dict) and layout.get("mountpoint") == "/"
        ),
        None,
    )
    if root is None or not root.get("disks"):
        return None
    sizes_by_disk: dict[str, int] = {}
    for config in (device.get("storageLayouts") or {}).get("configs") or []:
        size = config.get("size_per_disk")
        if not isinstance(size, int) or size <= 0:
            continue
        for disk in config.get("disks") or []:
            for key in ("wwn", "name", "serial"):
                if disk.get(key):
                    sizes_by_disk[str(disk[key])] = size
    sizes = [sizes_by_disk.get(str(disk)) for disk in root["disks"]]
    if not sizes or any(s is None for s in sizes):
        return None
    known_sizes = cast(list[int], sizes)
    # LVM and RAID 0 add the disks up; any other RAID level is at least the smallest disk.
    if len(known_sizes) == 1 or root.get("config") in ("lvm", "raid0"):
        total = sum(known_sizes)
    else:
        total = min(known_sizes)
    return math.floor(total / 1024**3)
