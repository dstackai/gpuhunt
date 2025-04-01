import logging
import re
from dataclasses import dataclass
from typing import Optional

from nebius.aio.channel import Credentials
from nebius.api.nebius.compute.v1 import (
    ListPlatformsRequest,
    ListPlatformsResponse,
    PlatformServiceClient,
    Preset,
)
from nebius.api.nebius.iam.v1 import (
    ListProjectsRequest,
    ListTenantsRequest,
    ProjectServiceClient,
    TenantServiceClient,
)
from nebius.sdk import SDK

from gpuhunt._internal.models import AcceleratorVendor, QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)


@dataclass
class PlatformGPU:
    name: str
    memory_gib: int
    price_hour: float
    vendor: AcceleratorVendor = AcceleratorVendor.NVIDIA


@dataclass
class Platform:
    name: str
    gpu: Optional[PlatformGPU]
    cpu_price_hour: float
    memory_gib_price_hour: float


# Until Nebius provides a pricing API, prices are taken from
# https://docs.nebius.com/compute/resources/pricing
PLATFORMS = [
    Platform(
        # NVIDIA® H100 NVLink with Intel Sapphire Rapids
        name="gpu-h100-sxm",
        gpu=PlatformGPU(
            name="H100",
            memory_gib=80,
            price_hour=2.118,
        ),
        cpu_price_hour=0.012,
        memory_gib_price_hour=0.0032,
    ),
    Platform(
        # NVIDIA® H200 NVLink with Intel Sapphire Rapids
        name="gpu-h200-sxm",
        gpu=PlatformGPU(
            name="H200",
            memory_gib=141,
            price_hour=2.668,
        ),
        cpu_price_hour=0.012,
        memory_gib_price_hour=0.0032,
    ),
    Platform(
        # NVIDIA® L40S PCIe with Intel Ice Lake
        name="gpu-l40s-a",
        gpu=PlatformGPU(
            name="L40S",
            memory_gib=48,
            price_hour=1.35,
        ),
        cpu_price_hour=0.012,
        memory_gib_price_hour=0.0032,
    ),
    Platform(
        # NVIDIA® L40S PCIe with AMD Epyc Genoa
        name="gpu-l40s-d",
        gpu=PlatformGPU(
            name="L40S",
            memory_gib=48,
            price_hour=1.35,
        ),
        cpu_price_hour=0.01,
        memory_gib_price_hour=0.0032,
    ),
    Platform(
        # Non-GPU AMD EPYC Genoa
        name="cpu-d3",
        gpu=None,
        cpu_price_hour=0.012,
        memory_gib_price_hour=0.0032,
    ),
    Platform(
        # Non-GPU Intel Ice Lake
        name="cpu-e2",
        gpu=None,
        cpu_price_hour=0.012,
        memory_gib_price_hour=0.0032,
    ),
]
PLATFORMS_MAP = {p.name: p for p in PLATFORMS}
TIMEOUT = 7


class NebiusProvider(AbstractProvider):
    NAME = "nebius"

    def __init__(self, credentials: Credentials) -> None:
        self.credentials = credentials

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        items: list[RawCatalogItem] = []
        sdk = SDK(credentials=self.credentials)
        try:
            for region_code, region_name in get_regions_map(sdk).items():
                for platform in list_platforms(sdk, region_code).items:
                    known_platform_details = PLATFORMS_MAP.get(platform.metadata.name)
                    if known_platform_details is None:
                        logger.warning(f"Unknown platform: {platform.metadata.name}")
                        continue
                    for preset in platform.spec.presets:
                        items.append(make_item(known_platform_details, preset, region_name))
        finally:
            sdk.sync_close(timeout=TIMEOUT)
        items.sort(key=lambda i: i.price)
        return items


def get_regions_map(sdk: SDK) -> dict[str, str]:
    """
    Returns:
        `{"e00": "eu-north1", "e01": "eu-west1", ...}`
    """
    tenants = TenantServiceClient(sdk).list(ListTenantsRequest(), timeout=TIMEOUT).wait()
    if len(tenants.items) != 1:
        raise ValueError(f"Expected to find 1 tenant, found {(len(tenants.items))}")
    projects = (
        ProjectServiceClient(sdk)
        .list(ListProjectsRequest(parent_id=tenants.items[0].metadata.id), timeout=TIMEOUT)
        .wait()
    )
    result = {}
    for project in projects.items:
        match = re.match(r"^project-([a-z]\d\d)", project.metadata.id)
        if match is None:
            logger.error(f"Could not parse project id {project.metadata.id!r}")
            continue
        result[match.group(1)] = project.status.region
    return result


def list_platforms(sdk: SDK, region_code: str) -> ListPlatformsResponse:
    req = ListPlatformsRequest(
        page_size=999,
        parent_id=f"project-{region_code}public-images",
    )
    return PlatformServiceClient(sdk).list(req, timeout=TIMEOUT).wait()


def make_item(platform: Platform, preset: Preset, region: str) -> RawCatalogItem:
    item = RawCatalogItem(
        instance_name=f"{platform.name} {preset.name}",
        location=region,
        price=(
            preset.resources.vcpu_count * platform.cpu_price_hour
            + preset.resources.memory_gibibytes * platform.memory_gib_price_hour
        ),
        cpu=preset.resources.vcpu_count,
        memory=preset.resources.memory_gibibytes,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        gpu_vendor=None,
        spot=False,
        disk_size=None,
    )
    if platform.gpu is not None:
        item.gpu_count = preset.resources.gpu_count
        item.gpu_name = platform.gpu.name
        item.gpu_memory = platform.gpu.memory_gib
        item.gpu_vendor = platform.gpu.vendor.value
        item.price += item.gpu_count * platform.gpu.price_hour
    item.price = round(item.price, 8)  # fix floating point precision errors
    return item
