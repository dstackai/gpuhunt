import copy
import csv
import datetime
import logging
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import boto3
import requests
from botocore.exceptions import ClientError, EndpointConnectionError

from gpuhunt._internal.models import QueryFilter, RawCatalogItem
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
ec2_pricing_url = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.csv"
)
disclaimer_rows_skip = 5
# https://aws.amazon.com/ec2/previous-generation/
previous_generation_families = [
    "a1.",
    "c1.",
    "c3.",
    "c4.",
    "g2.",
    "g3.",
    "g3s.",
    "i2.",
    "m1.",
    "m2.",
    "m3.",
    "p2.",
    "r3.",
    "r4.",
    "t1.",
    "cr1.",
    "hs1.",
]
pricing_filters = {
    "TermType": ["OnDemand"],
    "Tenancy": ["Shared"],
    "Operating System": ["Linux"],
    "CapacityStatus": ["Used"],
    "Unit": ["Hrs"],
    "Currency": ["USD"],
    "Pre Installed S/W": ["", "NA"],
    "MarketOption": ["OnDemand"],
}
describe_instances_limit = 100


class AWSProvider(AbstractProvider):
    """
    AWSProvider parses Bulk API index file for AmazonEC2 in all regions and fills missing GPU details

    Required IAM permissions:
    * `ec2:DescribeInstanceTypes`
    """

    NAME = "aws"

    def __init__(self, cache_path: Optional[str] = None):
        if cache_path:
            self.cache_path = cache_path
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.cache_path = self.temp_dir.name + "/index.csv"
        # todo aws creds
        self.preview_gpus = {
            "p4de.24xlarge": ("A100", 80.0),
        }

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> list[RawCatalogItem]:
        if not os.path.exists(self.cache_path):
            logger.info("Downloading EC2 prices to %s", self.cache_path)
            with requests.get(ec2_pricing_url, stream=True, timeout=20) as r:
                r.raise_for_status()
                with open(self.cache_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        offers = []
        with open(self.cache_path, newline="") as f:
            for _ in range(disclaimer_rows_skip):
                f.readline()
            reader: Iterable[dict[str, str]] = csv.DictReader(f)
            for row in reader:
                if self.skip(row):
                    continue
                gpu_count = _parse_gpu_count(row["GPU"])
                if gpu_count is None:
                    continue
                offer = RawCatalogItem(
                    instance_name=row["Instance Type"],
                    location=row["Region Code"],
                    price=float(row["PricePerUnit"]),
                    cpu=int(row["vCPU"]),
                    memory=_parse_memory(row["Memory"]),
                    gpu_vendor=None,
                    gpu_count=gpu_count,
                    spot=False,
                    gpu_name=None,
                    gpu_memory=None,
                    disk_size=None,
                )
                offers.append(offer)
        self.fill_gpu_details(offers)
        offers = self.add_spots(offers)
        return sorted(offers, key=lambda i: i.price)

    def skip(self, row: dict[str, str]) -> bool:
        if any(row["Instance Type"].startswith(family) for family in previous_generation_families):
            return True
        for key, values in pricing_filters.items():
            if row[key] not in values:
                return True
        return False

    def fill_gpu_details(self, offers: list[RawCatalogItem]):
        regions = defaultdict(list)
        for offer in offers:
            if offer.gpu_count > 0 and offer.instance_name not in self.preview_gpus:
                regions[offer.location].append(offer.instance_name)

        gpus = copy.deepcopy(self.preview_gpus)
        while regions:
            region = max(regions, key=lambda r: len(regions[r]))
            instance_types = regions.pop(region)

            client = boto3.client("ec2", region_name=region)
            paginator = client.get_paginator("describe_instance_types")
            for offset in range(0, len(instance_types), describe_instances_limit):
                logger.info("Fetching GPU details for %s (offset=%s)", region, offset)
                pages = paginator.paginate(
                    InstanceTypes=instance_types[offset : offset + describe_instances_limit]
                )
                for page in pages:
                    for i in page["InstanceTypes"]:
                        if "GpuInfo" in i:
                            gpu = i["GpuInfo"]["Gpus"][0]
                            gpus[i["InstanceType"]] = (
                                gpu["Name"],
                                _get_gpu_memory_gib(gpu["Name"], gpu["MemoryInfo"]["SizeInMiB"]),
                            )

            regions = {
                region: left
                for region, names in regions.items()
                if (left := [i for i in names if i not in instance_types])
            }

        for offer in offers:
            if offer.gpu_count > 0:
                if offer.instance_name in gpus:
                    offer.gpu_name, offer.gpu_memory = gpus[offer.instance_name]
                else:
                    logger.warning(
                        "GPU info not available for instance type %s, skipping GPU details",
                        offer.instance_name,
                    )
                    offer.gpu_count = 0

    def _add_spots_worker(
        self, region: str, instance_types: set[str]
    ) -> dict[tuple[str, str], float]:
        spot_prices = dict()
        logger.info("Fetching spot prices for %s", region)
        try:
            client = boto3.client("ec2", region_name=region)  # todo creds
            pages = client.get_paginator("describe_spot_price_history").paginate(
                Filters=[
                    {
                        "Name": "product-description",
                        "Values": ["Linux/UNIX"],
                    }
                ],
                InstanceTypes=list(instance_types),
                StartTime=datetime.datetime.utcnow(),
            )

            instance_prices = defaultdict(list)
            for page in pages:
                for item in page["SpotPriceHistory"]:
                    instance_prices[item["InstanceType"]].append(float(item["SpotPrice"]))
            for (
                instance_type,
                zone_prices,
            ) in instance_prices.items():  # reduce zone prices to a single value
                spot_prices[(instance_type, region)] = min(zone_prices)
        except (ClientError, EndpointConnectionError):
            return {}
        return spot_prices

    def add_spots(self, offers: list[RawCatalogItem]) -> list[RawCatalogItem]:
        region_instances = defaultdict(set)
        for offer in offers:
            region_instances[offer.location].add(offer.instance_name)

        spot_prices = dict()
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_region = {}
            for region, instance_types in region_instances.items():
                future = executor.submit(self._add_spots_worker, region, instance_types)
                future_to_region[future] = region
            for future in as_completed(future_to_region):
                spot_prices.update(future.result())

        spot_offers = []
        for offer in offers:
            if (price := spot_prices.get((offer.instance_name, offer.location))) is None:
                continue
            spot_offer = copy.deepcopy(offer)
            spot_offer.spot = True
            spot_offer.price = price
            spot_offers.append(spot_offer)
        return offers + spot_offers

    @classmethod
    def filter(cls, offers: list[RawCatalogItem]) -> list[RawCatalogItem]:
        return [
            i
            for i in offers
            if any(
                i.instance_name.startswith(family)
                for family in [
                    "m7i.",
                    "c7i.",
                    "r7i.",
                    "t3.",
                    "t2.small",
                    "c5.",
                    "m5.",
                    "p5.",
                    "p5e.",
                    "p4d.",
                    "p4de.",
                    "p3.",
                    "g6.",
                    "g6e.",
                    "gr6.",
                    "g5.",
                    "g4dn.",
                ]
            )
        ]


def _get_gpu_memory_gib(gpu_name: str, reported_memory_mib: int) -> float:
    """
    Fixes L4 memory size misreported by AWS API
    """

    if gpu_name != "L4":
        return reported_memory_mib / 1024

    if reported_memory_mib not in (22888, 91553, 183105):
        logger.warning(
            "The L4 memory size reported by AWS changed. "
            "Please check that it is now correct and remove the hardcoded size if it is."
        )
    return 24


def _parse_memory(s: str) -> float:
    r = re.match(r"^([0-9.]+) GiB$", s)
    return float(r.group(1))


def _parse_gpu_count(s: str) -> Optional[int]:
    if not s:
        return 0
    count = float(s)
    if count < 1:
        # AWS fractional GPUs not supported
        return None
    return int(count)
