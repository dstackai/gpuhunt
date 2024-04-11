import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests

from gpuhunt._internal.constraints import KNOWN_GPUS
from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
bundles_url = "https://console.vast.ai/api/v0/bundles/"
kilo = 1000
Operators = Literal["lt", "lte", "eq", "gte", "gt"]
FilterValue = Union[int, float, str, bool]


class VastAIProvider(AbstractProvider):
    NAME = "vastai"

    def __init__(self, extra_filters: Optional[Dict[str, Dict[Operators, FilterValue]]] = None):
        self.extra_filters = extra_filters

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        filters: Dict[str, Any] = self.make_filters(query_filter or QueryFilter())
        if self.extra_filters:
            for key, constraints in self.extra_filters.items():
                for op, value in constraints.items():
                    filters[key][op] = value
        filters["rentable"]["eq"] = True
        filters["rented"]["eq"] = False
        filters["order"] = [["score", "desc"]]
        resp = requests.post(bundles_url, json=filters)
        resp.raise_for_status()
        data = resp.json()

        instance_offers = []
        for offer in data["offers"]:
            disk_size = query_filter and query_filter.min_disk_size or offer["disk_space"]
            if not self.satisfies_filters(offer, filters):
                logger.warning("Offer %s does not satisfy filters", offer["id"])
                continue
            gpu_name = get_gpu_name(offer["gpu_name"])
            gpu_memory = normalize_gpu_memory(gpu_name, offer["gpu_ram"])
            ondemand_offer = RawCatalogItem(
                instance_name=str(offer["id"]),
                location=get_location(offer["geolocation"]),
                # storage_cost is $/gb/month
                price=round(
                    offer["dph_base"] + disk_size * offer["storage_cost"] / 30 / 24,
                    5,
                ),
                cpu=int(offer["cpu_cores_effective"]),
                memory=float(
                    int(
                        offer["cpu_ram"] * offer["cpu_cores_effective"] / offer["cpu_cores"] / kilo
                    )
                ),
                gpu_count=offer["num_gpus"],
                gpu_name=gpu_name,
                gpu_memory=float(gpu_memory),
                spot=False,
                disk_size=disk_size,
            )
            instance_offers.append(ondemand_offer)

            spot_offer = copy.deepcopy(ondemand_offer)
            spot_offer.price = round(offer["min_bid"], 5)
            spot_offer.spot = True
            instance_offers.append(spot_offer)
        return instance_offers

    @staticmethod
    def make_filters(q: QueryFilter) -> Dict[str, Dict[Operators, FilterValue]]:
        filters = defaultdict(dict)
        if q.min_cpu is not None:
            filters["cpu_cores"]["gte"] = q.min_cpu
        if q.max_cpu is not None:
            filters["cpu_cores"]["lte"] = q.max_cpu
        if q.min_memory is not None:
            filters["cpu_ram"]["gte"] = q.min_memory * kilo
        if q.max_memory is not None:
            filters["cpu_ram"]["lte"] = q.max_memory * kilo
        if q.min_gpu_count is not None:
            filters["num_gpus"]["gte"] = q.min_gpu_count
        if q.max_gpu_count is not None:
            filters["num_gpus"]["lte"] = q.max_gpu_count
        # We cannot reliably filter by GPU memory, because it is not the same for a specific GPU model
        if q.min_disk_size is not None:
            filters["disk_space"]["gte"] = q.min_disk_size
        if q.max_disk_size is not None:
            filters["disk_space"]["lte"] = q.max_disk_size
        if q.min_price is not None:
            filters["dph_total"]["gte"] = q.min_price
        if q.max_price is not None:
            filters["dph_total"]["lte"] = q.max_price
        # TODO(egor-s): add compute capability info for all GPUs
        if q.min_compute_capability is not None:
            filters["compute_capability"]["gte"] = compute_cap(q.min_compute_capability)
        if q.max_compute_capability is not None:
            filters["compute_capability"]["lte"] = compute_cap(q.max_compute_capability)
        return filters

    @staticmethod
    def satisfies_filters(offer: dict, filters: Dict[str, Dict[Operators, FilterValue]]) -> bool:
        for key in filters:
            if key not in offer:
                continue
            for op, value in filters[key].items():
                if op == "lt" and offer[key] >= value:
                    return False
                if op == "lte" and offer[key] > value:
                    return False
                if op == "eq" and offer[key] != value:
                    return False
                if op == "gte" and offer[key] < value:
                    return False
                if op == "gt" and offer[key] <= value:
                    return False
        return True


def get_gpu_name(gpu_name: str) -> str:
    gpu_name = gpu_name.replace("RTX A", "A").replace("Tesla ", "").replace("Q ", "")
    if gpu_name.startswith("A100 "):
        return "A100"
    return gpu_name.replace(" ", "")


def normalize_gpu_memory(gpu_name: str, memory_mib: float) -> int:
    known_memory = [gpu.memory for gpu in KNOWN_GPUS if gpu.name == gpu_name]
    if known_memory:
        # return the closest known value
        return min(known_memory, key=lambda x: abs(x - memory_mib / kilo))
    return int(memory_mib / kilo)


def get_location(location: Optional[str]) -> str:
    if location is None:
        return ""
    try:
        city, country = location.replace(", ", ",").split(",")
        location = f"{country}-{city}"
    except ValueError:
        pass
    return location.lower().replace(" ", "")


def compute_cap(cc: Tuple[int, int]) -> str:
    """
    >>> compute_cap((7, 0))
    '700'
    >>> compute_cap((7, 5))
    '750'
    """
    major, minor = cc
    return f"{major}{str(minor).ljust(2, '0')}"
