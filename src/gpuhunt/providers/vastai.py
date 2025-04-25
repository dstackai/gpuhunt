import copy
import logging
import re
from collections import defaultdict
from typing import Any, Literal, Optional, Union

import requests

from gpuhunt._internal.constraints import correct_gpu_memory_gib
from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
bundles_url = "https://console.vast.ai/api/v0/bundles/"
kilo = 1000
Operators = Literal["lt", "lte", "eq", "gte", "gt"]
FilterValue = Union[int, float, str, bool]


class VastAIProvider(AbstractProvider):
    NAME = "vastai"

    def __init__(self, extra_filters: Optional[dict[str, dict[Operators, FilterValue]]] = None):
        self.extra_filters = extra_filters

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        filters: dict[str, Any] = self.make_filters(query_filter or QueryFilter())
        if self.extra_filters:
            for key, constraints in self.extra_filters.items():
                for op, value in constraints.items():
                    filters[key][op] = value
        filters["rentable"]["eq"] = True
        filters["rented"]["eq"] = False
        filters["order"] = [["score", "desc"]]
        resp = requests.post(bundles_url, json=filters, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        instance_offers = []
        for offer in data["offers"]:
            cpu_cores = offer["cpu_cores"]
            # although this is not stated in the docs, the value can be None
            if cpu_cores:
                memory = float(
                    int(offer["cpu_ram"] * offer["cpu_cores_effective"] / cpu_cores / kilo)
                )
            else:
                memory = 0.0
            disk_size = query_filter and query_filter.min_disk_size or offer["disk_space"]
            if not self.satisfies_filters(offer, filters):
                logger.warning("Offer %s does not satisfy filters", offer["id"])
                continue
            gpu_name = get_gpu_name(offer["gpu_name"])
            gpu_memory = correct_gpu_memory_gib(gpu_name, offer["gpu_ram"])
            ondemand_offer = RawCatalogItem(
                instance_name=str(offer["id"]),
                location=get_location(offer["geolocation"]),
                # storage_cost is $/gb/month
                price=round(
                    offer["dph_base"] + disk_size * offer["storage_cost"] / 30 / 24,
                    5,
                ),
                cpu=int(offer["cpu_cores_effective"]),
                memory=memory,
                gpu_vendor=None,
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
    def make_filters(q: QueryFilter) -> dict[str, dict[Operators, FilterValue]]:
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
        if q.gpu_name:
            vastai_gpu_names = []
            for g in q.gpu_name:
                vastai_gpu_names.extend(get_vastai_gpu_names(g))
            filters["gpu_name"]["in"] = vastai_gpu_names
        # See correct_gpu_memory_gib in gpuhunt/_internal/constraints.py
        if q.min_gpu_memory is not None:
            filters["gpu_ram"]["gte"] = q.min_gpu_memory * 1024 * 0.93
        if q.max_gpu_memory is not None:
            filters["gpu_ram"]["lte"] = q.max_gpu_memory * 1024 * 1.07
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
    def satisfies_filters(offer: dict, filters: dict[str, dict[Operators, FilterValue]]) -> bool:
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


def get_vastai_gpu_names(gpu_name: str) -> list[str]:
    gpu_name = re.sub(r"^RTX(\d)", r"RTX \1", gpu_name)  # RTX4090 -> RTX 4090
    if gpu_name == "A100":
        return ["A100 PCIE", "A100 SXM4"]
    if gpu_name == "H100":
        return ["H100 SXM"]
    gpu_name = re.sub(r"^RTX(\d{4})", r"RTX \1", gpu_name)  # A5000 -> RTX A5000
    gpu_name = re.sub(r"NVL$", " NVL", gpu_name)  # H100NVL -> H100 NVL
    return [gpu_name]


def get_gpu_name(gpu_name: str) -> str:
    gpu_name = gpu_name.replace("RTX A", "A").replace("Tesla ", "").replace("Q ", "")
    if gpu_name.startswith("A100 "):
        return "A100"
    if gpu_name.startswith("H100 ") and "NVL" not in gpu_name:
        return "H100"
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


def compute_cap(cc: tuple[int, int]) -> str:
    """
    >>> compute_cap((7, 0))
    '700'
    >>> compute_cap((7, 5))
    '750'
    """
    major, minor = cc
    return f"{major}{str(minor).ljust(2, '0')}"
