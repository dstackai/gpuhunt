import csv
from typing import Iterable

from gpuhunt._models import InstanceOffer


def dump(offers: list[InstanceOffer], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(InstanceOffer.model_fields.keys()))
        writer.writeheader()
        for offer in offers:
            writer.writerow(offer.model_dump())


def load(path: str) -> list[InstanceOffer]:
    offers = []
    with open(path, "r", newline="") as f:
        reader: Iterable[dict[str, str]] = csv.DictReader(f)
        for row in reader:
            offer = InstanceOffer.model_validate(row)
            offers.append(offer)
    return offers


def sort_key(offer: InstanceOffer):
    return offer.gpu_count, offer.instance_name, offer.price, offer.location
