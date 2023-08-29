import re
from typing import Tuple

import bs4
import requests

from dstack.pricing.models import InstanceOffer
from dstack.pricing.providers import AbstractProvider

specs_page_url = "https://lambdalabs.com/service/gpu-cloud"
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
    def get(self) -> list[InstanceOffer]:
        soup = bs4.BeautifulSoup(requests.get(specs_page_url).text, "html.parser")
        table = soup.find("span", text="GPUs").find_parent("table")

        offers = []
        for row in table.find("tbody").find_all("tr"):
            cells = row.find_all("td")
            gpu_memory = parse_memory(cells[1].text)
            instance_name, gpu_count, gpu_name = parse_name(cells[0].text, gpu_memory)
            offer = InstanceOffer(
                instance_name=instance_name,
                price=parse_price(cells[5].text),
                cpu=int(cells[2].text),
                memory=parse_memory(cells[3].text),
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                spot=False,
            )
            offers.append(offer)
            if offer.gpu_name == "A100" and offer.gpu_count == 1:
                offer = offer.model_copy()
                offer.instance_name += "_sxm4"
                offers.append(offer)
        return self.add_regions(offers)

    def add_regions(self, offers: list[InstanceOffer]) -> list[InstanceOffer]:
        # TODO: we don't know which regions are actually available for each instance type
        region_offers = []
        for region in all_regions:
            for offer in offers:
                offer = offer.model_copy()
                offer.location = region
                region_offers.append(offer)
        return region_offers


def parse_memory(v: str) -> float:
    r = re.match(r"^([\d.]+) ?([GT])i?B$", v)
    value, unit = r.groups()
    value = float(value)
    if unit == "T":
        value *= 1000
    return value


def parse_price(v: str) -> float:
    r = re.match(r"^\$([\d.]+) / hr$", v)
    return float(r.group(1))


def parse_name(v: str, gpu_memory: float) -> Tuple[str, int, str]:
    """Returns instance name, number of GPUs, and GPU name"""
    v = v.replace("RTX 6000", "RTX6000")
    r = re.match(r"^(\d)x NVIDIA(?: RTX| Quadro| Tesla)? ([A-Z]+\d+)", v)
    count, gpu_name = r.groups()
    count = int(count)

    suffix = ""
    if gpu_name == "H100" and count == 1:
        suffix = "_pcie"
    if gpu_name == "H100" and count == 8:
        suffix = "_sxm5"
    if gpu_name == "A100" and count == 8 and int(gpu_memory) == 80:
        suffix = "_80gb_sxm4"
    return f"gpu_{count}x_{gpu_name.lower()}{suffix}", count, gpu_name
