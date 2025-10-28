import logging
import re
from dataclasses import dataclass
from typing import Optional, cast

from nebius.aio.channel import Credentials
from nebius.api.nebius.billing.v1alpha1 import (
    CalculatorServiceClient,
    EstimateRequest,
    ResourceSpec,
)
from nebius.api.nebius.common.v1 import ResourceMetadata
from nebius.api.nebius.compute.v1 import (
    CreateInstanceRequest,
    InstanceSpec,
    ListPlatformsRequest,
    ListPlatformsResponse,
    PlatformServiceClient,
    PreemptibleSpec,
    Preset,
    ResourcesSpec,
)
from nebius.api.nebius.iam.v1 import (
    ListProjectsRequest,
    ListTenantsRequest,
    ProjectServiceClient,
    TenantServiceClient,
)
from nebius.sdk import SDK
from typing_extensions import TypedDict

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import (
    AcceleratorInfo,
    AcceleratorVendor,
    JSONObject,
    QueryFilter,
    RawCatalogItem,
)
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
TIMEOUT = 7


@dataclass(frozen=True)
class InfinibandFabric:
    name: str
    platform: str
    region: str


# https://docs.nebius.com/compute/clusters/gpu#fabrics
INFINIBAND_FABRICS = [
    InfinibandFabric("fabric-2", "gpu-h100-sxm", "eu-north1"),
    InfinibandFabric("fabric-3", "gpu-h100-sxm", "eu-north1"),
    InfinibandFabric("fabric-4", "gpu-h100-sxm", "eu-north1"),
    InfinibandFabric("fabric-5", "gpu-h200-sxm", "eu-west1"),
    InfinibandFabric("fabric-6", "gpu-h100-sxm", "eu-north1"),
    InfinibandFabric("fabric-7", "gpu-h200-sxm", "eu-north1"),
    InfinibandFabric("us-central1-a", "gpu-h200-sxm", "us-central1"),
    InfinibandFabric("us-central1-b", "gpu-b200-sxm", "us-central1"),
]


class NebiusProvider(AbstractProvider):
    NAME = "nebius"

    def __init__(self, credentials: Credentials) -> None:
        self.credentials = credentials

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        items: list[RawCatalogItem] = []
        sdk = SDK(credentials=self.credentials)
        calculator = CalculatorServiceClient(sdk)
        try:
            region_to_project_id = get_sample_projects(sdk)
            for region, project_id in region_to_project_id.items():
                platforms = list_platforms(sdk, project_id).items
                for platform in platforms:
                    logger.info("Processing %s/%s", region, platform.metadata.name)
                    gpu = get_gpu_info(platform.metadata.name)
                    for preset in platform.spec.presets:
                        for spot in [False] + (
                            [True] if platform.status.allowed_for_preemptibles else []
                        ):
                            price = get_price(
                                calculator, project_id, platform.metadata.name, preset.name, spot
                            )
                            item = make_item(
                                platform.metadata.name, preset, gpu, region, spot, price
                            )
                            if item is not None:
                                items.append(item)
        finally:
            sdk.sync_close(timeout=TIMEOUT)
        items.sort(key=lambda i: i.price)
        return items


class NebiusCatalogItemProviderData(TypedDict):
    fabrics: list[str]


def get_sample_projects(sdk: SDK) -> dict[str, str]:
    """
    Returns:
        A mapping from region names to project IDs of random projects in those regions.
    """
    tenants = TenantServiceClient(sdk).list(ListTenantsRequest(), per_retry_timeout=TIMEOUT).wait()
    if len(tenants.items) != 1:
        raise ValueError(f"Expected to find 1 tenant, found {(len(tenants.items))}")
    projects = (
        ProjectServiceClient(sdk)
        .list(
            ListProjectsRequest(parent_id=tenants.items[0].metadata.id), per_retry_timeout=TIMEOUT
        )
        .wait()
    )
    region_to_project_id = {}
    for project in projects.items:
        region_to_project_id[project.status.region] = project.metadata.id
    return region_to_project_id


def get_price(
    calculator: CalculatorServiceClient, project_id: str, platform: str, preset: str, spot: bool
) -> float:
    spec = CreateInstanceRequest(
        metadata=ResourceMetadata(parent_id=project_id),
        spec=InstanceSpec(
            resources=ResourcesSpec(platform=platform, preset=preset),
            preemptible=PreemptibleSpec() if spot else None,
        ),
    )
    estimate = calculator.estimate(
        request=EstimateRequest(resource_spec=ResourceSpec(compute_instance_spec=spec))
    ).wait()
    return float(estimate.hourly_cost.general.total.cost)


def list_platforms(sdk: SDK, project_id: str) -> ListPlatformsResponse:
    req = ListPlatformsRequest(
        page_size=999,
        parent_id=project_id,
    )
    return PlatformServiceClient(sdk).list(req, per_retry_timeout=TIMEOUT).wait()


def get_gpu_info(platform: str) -> Optional[AcceleratorInfo]:
    m = re.match(r"gpu-([^-]+)-", platform)
    if m is None:
        return None
    gpu_name = m.group(1)
    accelerator_info = find_accelerators(names=[gpu_name], vendors=[AcceleratorVendor.NVIDIA])
    if len(accelerator_info) != 1:
        return None
    return accelerator_info[0]


def make_item(
    platform: str,
    preset: Preset,
    gpu: Optional[AcceleratorInfo],
    region: str,
    spot: bool,
    price: float,
) -> Optional[RawCatalogItem]:
    fabrics = []
    if preset.allow_gpu_clustering:
        fabrics = [
            f.name for f in INFINIBAND_FABRICS if f.platform == platform and f.region == region
        ]

    item = RawCatalogItem(
        instance_name=f"{platform} {preset.name}",
        location=region,
        price=price,
        cpu=preset.resources.vcpu_count,
        memory=preset.resources.memory_gibibytes,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        gpu_vendor=None,
        spot=spot,
        disk_size=None,
        provider_data=cast(JSONObject, NebiusCatalogItemProviderData(fabrics=fabrics)),
    )

    if preset.resources.gpu_count:
        if gpu is None:
            logger.warning(
                "Platform %s preset %s has GPUs, but they could not be identified. Skipping",
                platform,
                preset.name,
            )
            return None
        item.gpu_count = preset.resources.gpu_count
        item.gpu_name = gpu.name
        item.gpu_memory = float(gpu.memory)
        item.gpu_vendor = gpu.vendor

    return item
