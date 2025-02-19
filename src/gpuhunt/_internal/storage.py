import csv
import dataclasses
from typing import TypeVar

from gpuhunt._internal.models import RawCatalogItem

CATALOG_V1_FIELDS = [
    "instance_name",
    "location",
    "price",
    "cpu",
    "memory",
    "gpu_count",
    "gpu_name",
    "gpu_memory",
    "spot",
    "disk_size",
    "gpu_vendor",
]
T = TypeVar("T", bound=RawCatalogItem)


def dump(items: list[T], path: str, *, cls: type[T] = RawCatalogItem):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(cls)])
        writer.writeheader()
        for item in items:
            writer.writerow(item.dict())


def convert_catalog_v2_to_v1(path_v2: str, path_v1: str) -> None:
    with open(path_v2) as f_v2, open(path_v1, "w") as f_v1:
        reader = csv.DictReader(f_v2)
        writer = csv.DictWriter(f_v1, fieldnames=CATALOG_V1_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            if not row.get("flags"):
                writer.writerow(row)
