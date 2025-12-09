import copy
import itertools
import logging
import re
from collections.abc import Iterable
from typing import Optional

from verda import VerdaClient
from verda.instance_types import InstanceType

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

AMD_RX7900XTX = "RX7900XTX"
ALL_AMD_GPUS = [
    AMD_RX7900XTX,
]


class VerdaProvider(AbstractProvider):
    NAME = "verda"

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.verda_client = VerdaClient(client_id, client_secret)

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        instance_types = self._get_instance_types()
        locations = self._get_locations()

        spots = (True, False)
        location_codes = [loc["code"] for loc in locations]
        instances = generate_instances(spots, location_codes, instance_types)

        return sorted(instances, key=lambda x: x.price)

    def _get_instance_types(self) -> list[InstanceType]:
        return self.verda_client.instance_types.get()

    def _get_locations(self) -> list[dict]:
        return self.verda_client.locations.get()

    @classmethod
    def filter(cls, offers: list[RawCatalogItem]) -> list[RawCatalogItem]:
        return [o for o in offers if o.gpu_name not in ALL_AMD_GPUS]  # skip AMD GPU


def generate_instances(
    spots: Iterable[bool], location_codes: Iterable[str], instance_types: Iterable[InstanceType]
) -> list[RawCatalogItem]:
    instances = []
    for spot, location, instance in itertools.product(spots, location_codes, instance_types):
        item = transform_instance(copy.copy(instance), spot, location)
        if item is None:
            continue
        instances.append(RawCatalogItem.from_dict(item))
    return instances


def transform_instance(instance: InstanceType, spot: bool, location: str) -> Optional[dict]:
    gpu_memory = None
    gpu_count = instance.gpu["number_of_gpus"]
    gpu_name = None

    if instance.gpu["number_of_gpus"]:
        gpu_memory = instance.gpu_memory["size_in_gigabytes"] / instance.gpu["number_of_gpus"]
        gpu_name = get_gpu_name(instance.gpu["description"])

    if gpu_count and gpu_name is None:
        logger.warning(
            "Failed to get GPU name from description: '%s'", instance.gpu["description"]
        )
        return None

    raw = dict(
        instance_name=instance.instance_type,
        location=location,
        spot=spot,
        price=instance.spot_price_per_hour if spot else instance.price_per_hour,
        cpu=instance.cpu["number_of_cores"],
        memory=instance.memory["size_in_gigabytes"],
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
    )
    return raw


GPU_MAP = {
    r"\d+x B200 SXM6 180GB": "B200",
    r"\d+x B300 SXM6 262GB": "B300",
    r"\d+x H200 SXM5 141GB": "H200",
    r"\d+x H100 SXM5 80GB": "H100",
    r"\d+x A100 SXM4 80GB": "A100",
    r"\d+x A100 SXM4 40GB": "A100",
    r"\d+x RTX6000 Ada 48GB": "RTX6000Ada",
    r"\d+x RTX 6000 Ada 48GB": "RTX6000Ada",
    r"\d+x RTX PRO 6000 96GB": "RTXPRO6000",
    r"\d+x RTX A6000 48GB": "A6000",
    r"\d+x Tesla V100 16GB": "V100",
    r"\d+x L40S 48GB": "L40S",
    r"\d+x AMD 7900XTX": AMD_RX7900XTX,
}


def get_gpu_name(name: str) -> Optional[str]:
    for regex, gpu_name in GPU_MAP.items():
        if re.fullmatch(regex, name):
            return gpu_name
    return None
