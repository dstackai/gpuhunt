import copy
from typing import List, Optional

import requests

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

bundles_url = "https://console.vast.ai/api/v0/bundles"


class VastAIProvider(AbstractProvider):
    NAME = "vastai"

    def get(self, query_filter: Optional[QueryFilter] = None) -> List[RawCatalogItem]:
        data = requests.get(bundles_url).json()
        instance_offers = []
        for offer in data["offers"]:
            gpu_name = get_gpu_name(offer["gpu_name"])
            ondemand_offer = RawCatalogItem(
                instance_name=f"{offer['host_id']}",
                location=get_location(offer["geolocation"]),
                price=round(offer["dph_total"], 5),
                cpu=offer["cpu_cores"],
                memory=round(offer["cpu_ram"] / 1024),
                gpu_count=offer["num_gpus"],
                gpu_name=gpu_name,
                gpu_memory=round(offer["gpu_ram"] / 1024),
                spot=False,
            )
            instance_offers.append(ondemand_offer)

            spot_offer = copy.deepcopy(ondemand_offer)
            spot_offer.price = round(offer["min_bid"], 5)
            spot_offer.spot = True
            instance_offers.append(spot_offer)
        return instance_offers


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
