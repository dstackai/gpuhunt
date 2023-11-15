import logging
from typing import Dict, List, Optional

from datacrunch import DataCrunchClient
from datacrunch.instance_types.instance_types import InstanceType

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider


class DataCrunchProvider(AbstractProvider):
    NAME = "datacrunch"

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.datacrunch_client = DataCrunchClient(client_id, client_secret)

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        instance_types = self._get_instance_types()
        instance_map = make_instance_map(instance_types)

        availabilities = []
        for spot in [True, False]:
            raw_availabilities = self._get_availabilities(spot)
            spot_availabilities = _make_availability_list(spot, raw_availabilities)
            availabilities.extend(spot_availabilities)

        available_list = _make_list_available_instances(availabilities, instance_map)

        items = [RawCatalogItem.from_dict(item) for item in available_list]
        return items

    def _get_instance_types(self) -> List[InstanceType]:
        return self.datacrunch_client.instance_types.get()

    def _get_availabilities(self, spot: bool) -> List[dict]:
        return self.datacrunch_client.instances.get_availabilities(is_spot=spot)


def make_instance_map(instance_types):
    instance_map = {
        instance.instance_type: transform_instance(instance) for instance in instance_types
    }
    return instance_map


def transform_instance(instance: InstanceType) -> dict:
    raw = dict(
        instance_name=instance.instance_type,
        price=instance.price_per_hour,
        cpu=instance.cpu["number_of_cores"],
        memory=instance.memory["size_in_gigabytes"],
        gpu_count=instance.gpu["number_of_gpus"],
        gpu_name=gpu_name(instance.gpu["description"]),
        gpu_memory=instance.gpu_memory["size_in_gigabytes"],
    )
    return raw


def _make_availability_list(spot: bool, availabilities: List[dict]) -> List[tuple]:
    result = []
    for location in availabilities:
        for instance_type in location["availabilities"]:
            result.append((location["location_code"], spot, instance_type))
    return result


def _make_list_available_instances(availability_list, instances: Dict[str, dict]) -> List[dict]:
    result = []
    for location, spot, instance_type in availability_list:
        instance = instances[instance_type].copy()
        instance["location"] = location
        instance["spot"] = spot
        result.append(instance)
    return result


def gpu_name(name: str) -> str | None:
    if not name:
        return None

    gpu_map = {
        "1x H100 SXM5 80GB": "H100",
        "2x H100 SXM5 80GB": "H100",
        "4x H100 SXM5 80GB": "H100",
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
        "1x NVidia RTX6000 Ada 48GB": "RTX6000",
        "2x NVidia RTX6000 Ada 48GB": "RTX6000",
        "4x NVidia RTX6000 Ada 48GB": "RTX6000",
        "8x NVidia RTX6000 Ada 48GB": "RTX6000",
        "1x NVidia RTX A6000 48GB": "A6000",
        "2x NVidia RTX A6000 48GB": "A6000",
        "4x NVidia RTX A6000 48GB": "A6000",
        "8x NVidia RTX A6000 48GB": "A6000",
        "1x NVidia Tesla V100 16GB": "V100",
        "2x NVidia Tesla V100 16GB": "V100",
        "4x NVidia Tesla V100 16GB": "V100",
        "8x NVidia Tesla V100 16GB": "V100",
    }

    result = gpu_map.get(name)

    if result is None:
        logging.warning(f"There is no {name!r} in gpu_map")

    return result
