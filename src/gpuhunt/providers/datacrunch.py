import copy
import itertools
import logging
from collections.abc import Iterable
from typing import Optional

from datacrunch import DataCrunchClient
from datacrunch.instance_types.instance_types import InstanceType

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

AMD_RX7900XTX = "RX7900XTX"
ALL_AMD_GPUS = [
    AMD_RX7900XTX,
]


class DataCrunchProvider(AbstractProvider):
    NAME = "datacrunch"

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.datacrunch_client = DataCrunchClient(client_id, client_secret)

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
        return self.datacrunch_client.instance_types.get()

    def _get_locations(self) -> list[dict]:
        return self.datacrunch_client.locations.get()

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
    gpu_memory = 0
    gpu_count = instance.gpu["number_of_gpus"]
    gpu_name = None

    if instance.gpu["number_of_gpus"]:
        gpu_memory = instance.gpu_memory["size_in_gigabytes"] / instance.gpu["number_of_gpus"]
        gpu_name = get_gpu_name(instance.gpu["description"])

    if gpu_count and gpu_name is None:
        logger.warning("Can't get GPU name from description: '%s'", instance.gpu["description"])
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
    "1x H100 SXM5 80GB": "H100",
    "2x H100 SXM5 80GB": "H100",
    "4x H100 SXM5 80GB": "H100",
    "8x H100 SXM5 80GB": "H100",
    "1x A100 SXM4 80GB": "A100",
    "2x A100 SXM4 80GB": "A100",
    "4x A100 SXM4 80GB": "A100",
    "8x A100 SXM4 80GB": "A100",
    "1x A100 SXM4 40GB": "A100",
    "2x A100 SXM4 40GB": "A100",
    "4x A100 SXM4 40GB": "A100",
    "8x A100 SXM4 40GB": "A100",
    "1x NVIDIA RTX6000 Ada 48GB": "RTX6000Ada",
    "2x NVIDIA RTX6000 Ada 48GB": "RTX6000Ada",
    "4x NVIDIA RTX6000 Ada 48GB": "RTX6000Ada",
    "8x NVIDIA RTX6000 Ada 48GB": "RTX6000Ada",
    "1x NVIDIA RTX A6000 48GB": "A6000",
    "2x NVIDIA RTX A6000 48GB": "A6000",
    "4x NVIDIA RTX A6000 48GB": "A6000",
    "8x NVIDIA RTX A6000 48GB": "A6000",
    "1x NVIDIA Tesla V100 16GB": "V100",
    "2x NVIDIA Tesla V100 16GB": "V100",
    "4x NVIDIA Tesla V100 16GB": "V100",
    "8x NVIDIA Tesla V100 16GB": "V100",
    "1x NVIDIA L40S 48GB": "L40S",
    "2x NVIDIA L40S 48GB": "L40S",
    "4x NVIDIA L40S 48GB": "L40S",
    "8x NVIDIA L40S 48GB": "L40S",
    "1x AMD 7900XTX": AMD_RX7900XTX,
    "2x AMD 7900XTX": AMD_RX7900XTX,
    "4x AMD 7900XTX": AMD_RX7900XTX,
    "8x AMD 7900XTX": AMD_RX7900XTX,
    "12x AMD 7900XTX": AMD_RX7900XTX,
}


def get_gpu_name(name: str) -> Optional[str]:
    if not name:
        return None
    result = GPU_MAP.get(name)
    return result
