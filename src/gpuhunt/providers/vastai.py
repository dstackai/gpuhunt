import copy
import logging
from collections import defaultdict
from typing import List, Optional, Tuple

import requests

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
bundles_url = "https://console.vast.ai/api/v0/bundles/"
kilo = 1000


class VastAIProvider(AbstractProvider):
    NAME = "vastai"

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        filters = self.make_filters(query_filter or QueryFilter())
        filters["rentable"]["eq"] = True
        filters["direct_port_count"]["gte"] = 1  # publicly accessible
        filters["reliability2"]["gte"] = 0.9
        resp = requests.post(bundles_url, json=filters)
        resp.raise_for_status()
        data = resp.json()

        instance_offers = []
        for offer in data["offers"]:
            gpu_name = get_gpu_name(offer["gpu_name"])
            ondemand_offer = RawCatalogItem(
                instance_name=str(offer["id"]),
                location=get_location(offer["geolocation"]),
                price=round(offer["dph_total"], 5),  # TODO(egor-s) add disk price
                cpu=int(offer["cpu_cores_effective"]),
                memory=float(
                    int(
                        offer["cpu_ram"] * offer["cpu_cores_effective"] / offer["cpu_cores"] / kilo
                    )
                ),
                gpu_count=offer["num_gpus"],
                gpu_name=gpu_name,
                gpu_memory=float(int(offer["gpu_ram"] / kilo)),
                spot=False,
            )
            instance_offers.append(ondemand_offer)

            spot_offer = copy.deepcopy(ondemand_offer)
            spot_offer.price = round(offer["min_bid"], 5)
            spot_offer.spot = True
            instance_offers.append(spot_offer)
        return instance_offers

    @staticmethod
    def make_filters(q: QueryFilter) -> dict:
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
        if q.min_gpu_memory is not None:
            filters["gpu_ram"]["gte"] = q.min_gpu_memory * kilo
        if q.max_gpu_memory is not None:
            filters["gpu_ram"]["lte"] = q.max_gpu_memory * kilo
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


def get_gpu_name(gpu_name: str) -> str:
    gpu_name = gpu_name.replace("RTX A", "A").replace("Tesla ", "").replace("Q ", "")
    if gpu_name.startswith("A100 "):
        return "A100"
    return gpu_name.replace(" ", "")


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
