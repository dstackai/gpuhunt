import csv
import dataclasses
from collections.abc import Iterable
from typing import TypeVar

from gpuhunt._internal.models import RawCatalogItem

T = TypeVar("T", bound=RawCatalogItem)


def dump(items: list[T], path: str, *, cls: type[T] = RawCatalogItem):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(cls)])
        writer.writeheader()
        for item in items:
            writer.writerow(item.dict())


def load(path: str, *, cls: type[T] = RawCatalogItem) -> list[T]:
    items = []
    with open(path, newline="") as f:
        reader: Iterable[dict[str, str]] = csv.DictReader(f)
        for row in reader:
            item = cls.from_dict(row)
            items.append(item)
    return items
