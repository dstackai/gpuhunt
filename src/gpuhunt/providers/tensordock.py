import logging
from math import ceil
from typing import List, Optional, TypeVar, Union

import requests

from gpuhunt._internal.constraints import get_compute_capability, is_between
from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)

# https://documenter.getpostman.com/view/20973002/2s8YzMYRDc#2b4a3db0-c162-438c-aae4-6a88afc96fdb
marketplace_hostnodes_url = "https://marketplace.tensordock.com/api/v0/client/deploy/hostnodes"
marketplace_gpus = {
    "a100-pcie-80gb": "A100",
    "geforcegtx1070-pcie-8gb": "GTX1070",
    "geforcertx3060-pcie-12gb": "RTX3060",
    "geforcertx3060ti-pcie-8gb": "RTX3060Ti",
    "geforcertx3060tilhr-pcie-8gb": "RTX3060TiLHR",
    "geforcertx3070-pcie-8gb": "RTX3070",
    "geforcertx3070ti-pcie-8gb": "RTX3070Ti",
    "geforcertx3080-pcie-10gb": "RTX3080",
    "geforcertx3080ti-pcie-12gb": "RTX3080Ti",
    "geforcertx3090-pcie-24gb": "RTX3090",
    "geforcertx4090-pcie-24gb": "RTX4090",
    "l40-pcie-48gb": "L40",
    "rtxa4000-pcie-16gb": "A4000",
    "rtxa5000-pcie-24gb": "A5000",
    "rtxa6000-pcie-48gb": "A6000",
    "v100-pcie-16gb": "V100",
}

RAM_PER_VRAM = 2
RAM_PER_CORE = 6
CPU_DIV = 2  # has to be even
RAM_DIV = 2  # has to be even


class TensorDockProvider(AbstractProvider):
    NAME = "tensordock"

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        logger.info("Fetching TensorDock offers")

        hostnodes = requests.get(marketplace_hostnodes_url).json()["hostnodes"]
        offers = []
        for hostnode, details in hostnodes.items():
            location = details["location"]["country"].lower().replace(" ", "")
            if query_filter is not None:
                offers += self.optimize_offers(
                    query_filter,
                    details["specs"],
                    hostnode,
                    location,
                    balance_resources=balance_resources,
                )
            else:  # pick maximum possible configuration
                for gpu_name, gpu in details["specs"]["gpu"].items():
                    if gpu["amount"] == 0:
                        continue
                    offers.append(
                        RawCatalogItem(
                            instance_name=hostnode,
                            location=location,
                            price=round(
                                sum(
                                    details["specs"][key]["price"]
                                    * details["specs"][key]["amount"]
                                    for key in ("cpu", "ram", "storage")
                                )
                                + gpu["amount"] * gpu["price"],
                                5,
                            ),
                            cpu=round_down(details["specs"]["cpu"]["amount"], 2),
                            memory=float(round_down(details["specs"]["ram"]["amount"], 2)),
                            gpu_count=gpu["amount"],
                            gpu_name=convert_gpu_name(gpu_name),
                            gpu_memory=float(gpu["vram"]),
                            spot=False,
                            disk_size=float(details["specs"]["storage"]["amount"]),
                        )
                    )
        return sorted(offers, key=lambda i: i.price)

    @staticmethod
    def optimize_offers(
        q: QueryFilter,
        specs: dict,
        instance_name: str,
        location: str,
        balance_resources: bool = True,
    ) -> List[RawCatalogItem]:
        """
        Picks the best offer for the given query filter
        Doesn't respect max values, additional filtering is required

        Args:
            q: query filter
            specs: hostnode specs
            instance_name: hostnode `instance_name`
            location: hostnode `location`
            balance_resources: if True, will override query filter min values
        """
        offers = []
        for gpu_model, gpu_info in specs["gpu"].items():
            # filter by single gpu characteristics
            if not is_between(gpu_info["vram"], q.min_gpu_memory, q.max_gpu_memory):
                continue
            gpu_name = convert_gpu_name(gpu_model)
            if q.gpu_name is not None and gpu_name.lower() not in q.gpu_name:
                continue
            cc = get_compute_capability(gpu_name)
            if not cc or not is_between(cc, q.min_compute_capability, q.max_compute_capability):
                continue

            for gpu_count in range(1, gpu_info["amount"] + 1):  # try all possible gpu counts
                if not is_between(gpu_count, q.min_gpu_count, q.max_gpu_count):
                    continue
                total_gpu_memory = gpu_count * gpu_info["vram"]
                if not is_between(
                    total_gpu_memory, q.min_total_gpu_memory, q.max_total_gpu_memory
                ):
                    continue

                # we can't take 100% of CPU/RAM/storage if we don't take all GPUs
                multiplier = 0.75 if gpu_count < gpu_info["amount"] else 1
                available_memory = round_down(multiplier * specs["ram"]["amount"], RAM_DIV)
                available_cpu = round_down(multiplier * specs["cpu"]["amount"], CPU_DIV)
                available_disk = int(multiplier * specs["storage"]["amount"])

                memory = None
                if q.min_memory is not None:
                    if q.min_memory > available_memory:
                        continue
                    memory = round_up(
                        max_none(
                            q.min_memory,
                            gpu_count,  # 1 GB per GPU at least
                            q.min_cpu,  # 1 GB per CPU at least
                        ),
                        RAM_DIV,
                    )
                if memory is None or balance_resources:
                    memory = max_none(
                        memory,
                        min_none(
                            available_memory,
                            round_up(RAM_PER_VRAM * total_gpu_memory, RAM_DIV),
                            round_down(q.max_memory, RAM_DIV),  # can be None
                        ),
                    )

                cpu = None
                if q.min_cpu is not None:
                    if q.min_cpu > available_cpu:
                        continue
                    # 1 CPU per GPU at least
                    cpu = round_up(max(q.min_cpu, gpu_count), CPU_DIV)
                if cpu is None or balance_resources:
                    cpu = max_none(
                        cpu,
                        min_none(
                            available_cpu,
                            round_up(ceil(memory / RAM_PER_CORE), CPU_DIV),
                            round_down(q.max_cpu, CPU_DIV),  # can be None
                        ),
                    )

                disk_size = None
                if q.min_disk_size is not None:
                    if q.min_disk_size > available_disk:
                        continue
                    disk_size = q.min_disk_size
                if disk_size is None or balance_resources:
                    disk_size = max_none(
                        disk_size,
                        min_none(
                            available_disk,
                            max(memory, total_gpu_memory),
                            q.max_disk_size,  # can be None
                        ),
                    )

                price = round(
                    memory * specs["ram"]["price"]
                    + cpu * specs["cpu"]["price"]
                    + disk_size * specs["storage"]["price"]
                    + gpu_count * gpu_info["price"],
                    5,
                )

                offer = RawCatalogItem(
                    instance_name=instance_name,
                    location=location,
                    price=price,
                    cpu=cpu,
                    memory=float(memory),
                    gpu_name=gpu_name,
                    gpu_count=gpu_count,
                    gpu_memory=float(gpu_info["vram"]),
                    spot=False,
                    disk_size=disk_size,
                )
                offers.append(offer)
                break  # stop increasing gpu count
        return offers


def round_up(value: Optional[Union[int, float]], step: int) -> Optional[int]:
    if value is None:
        return None
    return round_down(value + step - 1, step)


def round_down(value: Optional[Union[int, float]], step: int) -> Optional[int]:
    if value is None:
        return None
    return value // step * step


T = TypeVar("T", bound=Union[int, float])


def min_none(*args: Optional[T]) -> T:
    return min(v for v in args if v is not None)


def max_none(*args: Optional[T]) -> T:
    return max(v for v in args if v is not None)


def convert_gpu_name(model: str) -> str:
    """
    >>> convert_gpu_name("geforcegtx1070-pcie-8gb")
    'GTX1070'
    >>> convert_gpu_name("geforcertx1111ti-pcie-13gb")
    'RTX1111Ti'
    >>> convert_gpu_name("a100-pcie-40gb")
    'A100'
    """
    if model in marketplace_gpus:
        return marketplace_gpus[model]
    model = model.split("-")[0]
    prefix = "geforce"
    if model.startswith(prefix):
        model = model[len(prefix) :]
    return model.upper().replace("TI", "Ti")
