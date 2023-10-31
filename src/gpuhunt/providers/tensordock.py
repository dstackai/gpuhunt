import logging
from typing import List, Optional, Union

import requests

from gpuhunt._internal.constraints import get_compute_capability, is_between, optimize
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


class TensorDockProvider(AbstractProvider):
    NAME = "tensordock"

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        logger.info("Fetching TensorDock offers")
        hostnodes = requests.get(marketplace_hostnodes_url).json()["hostnodes"]
        offers = []
        for hostnode, details in hostnodes.items():
            location = details["location"]["country"].lower().replace(" ", "")
            if query_filter is not None:
                offers += self.optimize_offers(query_filter, details["specs"], hostnode, location)
            else:
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
                            cpu=details["specs"]["cpu"]["amount"],
                            memory=float(round_down(details["specs"]["ram"]["amount"], 2)),
                            gpu_count=gpu["amount"],
                            gpu_name=convert_gpu_name(gpu_name),
                            gpu_memory=float(gpu["vram"]),
                            spot=False,
                        )
                    )
        return offers

    @staticmethod
    def optimize_offers(
        q: QueryFilter, specs: dict, instance_name: str, location: str
    ) -> List[RawCatalogItem]:
        offers = []
        for gpu_model, gpu_info in specs["gpu"].items():
            # filter by single gpu characteristics
            if not is_between(gpu_info["vram"], q.min_gpu_memory, q.max_gpu_memory):
                continue
            gpu_name = convert_gpu_name(gpu_model)
            if q.gpu_name is not None and gpu_name.lower() not in q.gpu_name:
                continue
            if q.min_compute_capability is not None or q.max_compute_capability is not None:
                cc = get_compute_capability(gpu_name)
                if not cc or not is_between(
                    cc, q.min_compute_capability, q.max_compute_capability
                ):
                    continue

            for gpu_count in range(1, gpu_info["amount"] + 1):  # try all possible gpu counts
                if not is_between(gpu_count, q.min_gpu_count, q.max_gpu_count):
                    continue
                if not is_between(
                    gpu_count * gpu_info["vram"], q.min_total_gpu_memory, q.max_total_gpu_memory
                ):
                    continue
                # we can't take 100% of CPU/RAM/storage if we don't take all GPUs
                multiplier = 0.75 if gpu_count < gpu_info["amount"] else 1
                cpu = optimize(
                    int(multiplier * specs["cpu"]["amount"]),
                    q.min_cpu or 1,
                    q.max_cpu,
                )
                memory = optimize(  # has to be even
                    round_down(int(multiplier * specs["ram"]["amount"]), 2),
                    round_up(q.min_memory or 1, 2),
                    round_down(q.max_memory, 2) if q.max_memory is not None else None,
                )
                disk_size = optimize(  # 30 GB at least for Ubuntu
                    int(multiplier * specs["storage"]["amount"]),
                    q.min_disk_size or 30,
                    q.max_disk_size,
                )
                if cpu is None or memory is None or disk_size is None:
                    continue
                price = round(
                    cpu * specs["cpu"]["price"]
                    + memory * specs["ram"]["price"]
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
                )
                offers.append(offer)
                break  # stop increasing gpu count
        return offers


def round_up(value: Union[int, float], step: int) -> int:
    return round_down(value + step - 1, step)


def round_down(value: Union[int, float], step: int) -> int:
    return value // step * step


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
