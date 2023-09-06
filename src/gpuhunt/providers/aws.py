import copy
import csv
import datetime
import logging
import os
import re
import tempfile
from collections import defaultdict
from typing import Iterable, Optional

import boto3
import requests
from botocore.exceptions import ClientError, EndpointConnectionError

from gpuhunt._models import InstanceOffer
from gpuhunt.providers import AbstractProvider

logger = logging.getLogger(__name__)
ec2_pricing_url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.csv"
disclaimer_rows_skip = 5


class AWSProvider(AbstractProvider):
    """
    AWSProvider parses Bulk API index file for AmazonEC2 in all regions and fills missing GPU details

    Required IAM permissions:
    * `ec2:DescribeInstanceTypes`
    """

    def __init__(self, cache_path: Optional[str] = None):
        if cache_path:
            self.cache_path = cache_path
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.cache_path = self.temp_dir.name + "/index.csv"
        # todo aws creds
        self.filters = {
            "TermType": ["OnDemand"],
            "Tenancy": ["Shared"],
            "Operating System": ["Linux"],
            "CapacityStatus": ["Used"],
            "Unit": ["Hrs"],
            "Currency": ["USD"],
            "Pre Installed S/W": ["", "NA"],
        }
        self.preview_gpus = {
            "p4de.24xlarge": ("A100", 80.0),
        }

    def get(self) -> list[InstanceOffer]:
        if not os.path.exists(self.cache_path):
            logger.info("Downloading EC2 prices to %s", self.cache_path)
            with requests.get(ec2_pricing_url, stream=True) as r:
                r.raise_for_status()
                with open(self.cache_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        offers = []
        with open(self.cache_path, "r", newline="") as f:
            for _ in range(disclaimer_rows_skip):
                f.readline()
            reader: Iterable[dict[str, str]] = csv.DictReader(f)
            for row in reader:
                if self.skip(row):
                    continue
                offer = InstanceOffer(
                    instance_name=row["Instance Type"],
                    location=row["Region Code"],
                    price=float(row["PricePerUnit"]),
                    cpu=int(row["vCPU"]),
                    memory=parse_memory(row["Memory"]),
                    gpu_count=parse_optional_count(row["GPU"]),
                    spot=False,
                )
                offers.append(offer)
        self.fill_gpu_details(offers)
        return self.add_spots(offers)

    def skip(self, row: dict[str, str]) -> bool:
        for key, values in self.filters.items():
            if row[key] not in values:
                return True
        return False

    def fill_gpu_details(self, offers: list[InstanceOffer]):
        regions = defaultdict(list)
        for offer in offers:
            if offer.gpu_count > 0 and offer.instance_name not in self.preview_gpus:
                regions[offer.location].append(offer.instance_name)

        gpus = copy.deepcopy(self.preview_gpus)
        while regions:
            region = max(regions, key=lambda r: len(regions[r]))
            instance_types = regions.pop(region)

            logger.info("Fetching GPU details for %s", region)
            client = boto3.client("ec2", region_name=region)
            paginator = client.get_paginator("describe_instance_types")
            for page in paginator.paginate(InstanceTypes=instance_types):
                for i in page["InstanceTypes"]:
                    gpu = i["GpuInfo"]["Gpus"][0]
                    gpus[i["InstanceType"]] = (
                        gpu["Name"],
                        gpu["MemoryInfo"]["SizeInMiB"] / 1024,
                    )

            regions = {
                region: left
                for region, names in regions.items()
                if (left := [i for i in names if i not in instance_types])
            }

        for offer in offers:
            if offer.gpu_count > 0:
                offer.gpu_name, offer.gpu_memory = gpus[offer.instance_name]

    def add_spots(self, offers: list[InstanceOffer]) -> list[InstanceOffer]:
        region_instances = defaultdict(set)
        for offer in offers:
            region_instances[offer.location].add(offer.instance_name)

        spot_prices = dict()
        for region, instance_types in region_instances.items():
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
                        instance_prices[item["InstanceType"]].append(
                            float(item["SpotPrice"])
                        )
                for (
                    instance_type,
                    zone_prices,
                ) in instance_prices.items():  # reduce zone prices to a single value
                    spot_prices[(instance_type, region)] = min(zone_prices)
            except (ClientError, EndpointConnectionError) as e:
                pass

        spot_offers = []
        for offer in offers:
            if (
                price := spot_prices.get((offer.instance_name, offer.location))
            ) is None:
                continue
            spot_offer = copy.deepcopy(offer)
            spot_offer.spot = True
            spot_offer.price = price
            spot_offers.append(spot_offer)
        return offers + spot_offers

    @classmethod
    def filter(cls, offers: list[InstanceOffer]) -> list[InstanceOffer]:
        return [
            i
            for i in offers
            if any(
                i.instance_name.startswith(family)
                for family in [
                    "t2.small",
                    "c5.",
                    "m5.",
                    "p3.",
                    "g5.",
                    "g4dn.",
                    "p4d.",
                    "p4de.",
                ]
            )
        ]


def parse_memory(s: str) -> float:
    r = re.match(r"^([0-9.]+) GiB$", s)
    return float(r.group(1))


def parse_optional_count(s: str) -> int:
    if not s:
        return 0
    return int(s)
