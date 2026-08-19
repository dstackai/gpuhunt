import logging
import re

from requests import Session

from gpuhunt._internal.constraints import is_nvidia_superchip
from gpuhunt._internal.models import (
    AcceleratorVendor,
    CatalogItem,
    CPUArchitecture,
    QueryFilter,
)
from gpuhunt.providers.base import OfflineProvider

logger = logging.getLogger(__name__)
INSTANCE_TYPES_URL = "https://cloud.lambdalabs.com/api/v1/instance-types"
IMAGES_URL = "https://cloud.lambdalabs.com/api/v1/images"
TIMEOUT = 10

FLAG_ARM = "lambda-arm"


class LambdaLabsProvider(OfflineProvider):
    NAME = "lambdalabs"

    def __init__(self, token: str):
        self.session = Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def get(
        self,
        query_filter: QueryFilter | None = None,
        balance_resources: bool = True,
        apply_filter: bool = False,
    ) -> list[CatalogItem]:
        offers: list[CatalogItem] = []
        regions = self.list_regions()
        resp = self.session.get(INSTANCE_TYPES_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["data"]
        for instance in data.values():
            instance = instance["instance_type"]
            description = instance["description"]
            result = _parse_description(description)
            if result is None:
                logger.warning("Can't parse GPU info from description: %s", description)
                continue
            gpu_count, gpu_name, gpu_memory = result
            for region in regions:
                flags: list[str] = []
                cpu_arch = CPUArchitecture.X86
                if is_nvidia_superchip(gpu_name):
                    cpu_arch = CPUArchitecture.ARM
                    flags.append(FLAG_ARM)
                offer = CatalogItem(
                    provider=LambdaLabsProvider.NAME,
                    instance_name=instance["name"],
                    price=instance["price_cents_per_hour"] / 100,
                    cpu_arch=cpu_arch,
                    cpu=instance["specs"]["vcpus"],
                    memory=float(instance["specs"]["memory_gib"]) * 1.074,
                    gpu_vendor=AcceleratorVendor.NVIDIA if gpu_count else None,
                    gpu_count=gpu_count,
                    gpu_name=gpu_name,
                    gpu_memory=gpu_memory,
                    spot=False,
                    location=region,
                    disk_size=float(instance["specs"]["storage_gib"]) * 1.074,
                    flags=flags,
                )
                offers.append(offer)
        return sorted(offers, key=lambda i: i.price)

    def list_regions(self) -> list[str]:
        resp = self.session.get(IMAGES_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        regions = set()
        for image in resp.json()["data"]:
            regions.add(image["region"]["name"])
        return sorted(regions)


def _parse_description(v: str) -> tuple[int, str, float] | None:
    """Returns gpus count, gpu name, and GPU memory"""
    r = re.match(r"^(\d)x (?:Tesla )?(.+) \((\d+) GB", v)
    if r is None:
        return None
    count, gpu_name, gpu_memory = r.groups()
    return int(count), gpu_name.replace(" ", ""), float(gpu_memory)
