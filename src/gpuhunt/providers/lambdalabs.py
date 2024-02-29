import copy
import logging
import re
from typing import List, Optional, Tuple

import requests

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
instance_types_url = "https://cloud.lambdalabs.com/api/v1/instance-types"
all_regions = [
    "us-south-1",
    "us-west-2",
    "us-west-1",
    "us-midwest-1",
    "us-west-3",
    "us-east-1",
    "europe-central-1",
    "asia-south-1",
    "me-west-1",
    "asia-northeast-1",
    "asia-northeast-2",
]


class LambdaLabsProvider(AbstractProvider):
    NAME = "lambdalabs"

    def __init__(self, token: str):
        self.token = token

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        offers = []
        data = requests.get(
            instance_types_url, headers={"Authorization": f"Bearer {self.token}"}
        ).json()["data"]
        for instance in data.values():
            instance = instance["instance_type"]
            logger.info(instance["name"])
            description = instance["description"]
            result = parse_description(description)
            if result is None:
                logger.warning("Can't parse GPU info from description: %s", description)
                continue
            gpu_count, gpu_name, gpu_memory = result
            offer = RawCatalogItem(
                instance_name=instance["name"],
                price=instance["price_cents_per_hour"] / 100,
                cpu=instance["specs"]["vcpus"],
                memory=float(instance["specs"]["memory_gib"]) * 1.074,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                spot=False,
                location=None,
                disk_size=float(instance["specs"]["storage_gib"]) * 1.074,
            )
            offers.append(offer)
        offers = self.add_regions(offers)
        return sorted(offers, key=lambda i: i.price)

    def add_regions(self, offers: List[RawCatalogItem]) -> List[RawCatalogItem]:
        # TODO: we don't know which regions are actually available for each instance type
        region_offers = []
        for region in all_regions:
            for offer in offers:
                offer = copy.deepcopy(offer)
                offer.location = region
                region_offers.append(offer)
        return region_offers


def parse_description(v: str) -> Optional[Tuple[int, str, float]]:
    """Returns gpus count, gpu name, and GPU memory"""
    r = re.match(r"^(\d)x (?:Tesla )?(.+) \((\d+) GB", v)
    if r is None:
        return None
    count, gpu_name, gpu_memory = r.groups()
    return int(count), gpu_name.replace(" ", ""), float(gpu_memory)
