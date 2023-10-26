import logging
from typing import List, Optional, Union

import requests

from gpuhunt._internal.constraints import is_between, optimize
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
            location = (
                "-".join([details["location"][key] for key in ["country", "region", "city"]])
                .lower()
                .replace(" ", "")
            )
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
                            gpu_name=marketplace_gpus.get(gpu_name, gpu_name),
                            gpu_memory=float(gpu["vram"]),
                            spot=False,
                        )
                    )
        return offers

    @staticmethod
    def optimize_offers(
        q: QueryFilter, specs: dict, instance_name: str, location: str
    ) -> List[RawCatalogItem]:
        cpu = optimize(specs["cpu"]["amount"], q.min_cpu or 1, q.max_cpu)
        memory = optimize(  # has to be even
            round_down(specs["ram"]["amount"], 2),
            round_up(q.min_memory or 1, 2),
            round_down(q.max_memory, 2) if q.max_memory is not None else None,
        )
        disk_size = optimize(  # 30 GB at least for Ubuntu
            specs["storage"]["amount"],
            q.min_disk_size or 30,
            q.max_disk_size,
        )
        if cpu is None or memory is None or disk_size is None:
            return []
        base_price = sum(
            n * specs[key]["price"]
            for key, n in [("cpu", cpu), ("ram", memory), ("storage", disk_size)]
        )
        offers = []
        for gpu_name, gpu in specs["gpu"].items():
            gpu_name = marketplace_gpus.get(gpu_name, gpu_name)
            if q.gpu_name is not None and gpu_name not in q.gpu_name:
                continue
            if not is_between(gpu["vram"], q.min_gpu_memory, q.max_gpu_memory):
                continue
            if (
                gpu_count := optimize(gpu["amount"], q.min_gpu_count or 1, q.max_gpu_count)
            ) is None:
                continue
            # filter by total gpu memory
            if q.min_total_gpu_memory is None:
                min_total_gpu_memory = gpu_count * gpu["vram"]
            else:
                min_total_gpu_memory = max(q.min_total_gpu_memory, gpu_count * gpu["vram"])
            gpu_total_memory = optimize(
                gpu["amount"] * gpu["vram"],
                round_up(min_total_gpu_memory, gpu["vram"]),
                round_down(q.max_total_gpu_memory, gpu["vram"])
                if q.max_total_gpu_memory is not None
                else None,
            )
            if gpu_total_memory is None:
                continue
            gpu_count = gpu_total_memory // gpu["vram"]
            if not is_between(gpu_count, q.min_gpu_count, q.max_gpu_count):
                continue
            # make an offer
            offer = RawCatalogItem(
                instance_name=instance_name,
                location=location,
                price=round(gpu_count * gpu["price"] + base_price, 5),
                cpu=cpu,
                memory=float(memory),
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=float(gpu["vram"]),
                spot=False,
            )
            offers.append(offer)
        return offers


def round_up(value: Union[int, float], step: int) -> int:
    return round_down(value + step - 1, step)


def round_down(value: Union[int, float], step: int) -> int:
    return value // step * step
