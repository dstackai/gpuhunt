import csv
import dataclasses
from typing import Iterable, List, Type, TypeVar

from gpuhunt._internal.models import RawCatalogItem

T = TypeVar("T", bound=RawCatalogItem)


def dump(items: List[T], path: str, *, cls: Type[T] = RawCatalogItem):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(cls)])
        writer.writeheader()
        for item in items:
            writer.writerow(item.dict())


def load(path: str, *, cls: Type[T] = RawCatalogItem) -> List[T]:
    items = []
    with open(path, "r", newline="") as f:
        reader: Iterable[dict[str, str]] = csv.DictReader(f)
        for row in reader:
            item = cls.from_dict(row)
            items.append(item)
    return items
