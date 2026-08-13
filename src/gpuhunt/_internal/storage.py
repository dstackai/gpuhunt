import csv
import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import IO

from gpuhunt._internal.models import (
    AcceleratorVendor,
    CatalogItem,
    CPUArchitecture,
    bool_loader,
)
from gpuhunt._internal.utils import load_optional, load_required

logger = logging.getLogger(__name__)

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
# The columns of a v2 catalog file, in order. `provider` is not stored: the file name
# carries it. Listed explicitly rather than derived from `CatalogItem` so that adding a
# field to the model cannot change the published format.
CATALOG_V2_FIELDS = [
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
    "flags",
    "cpu_arch",
    "provider_data",
]


def item_to_row(item: CatalogItem) -> dict[str, str]:
    return {
        "instance_name": item.instance_name,
        "location": item.location,
        "price": str(item.price),
        "cpu": str(item.cpu),
        "memory": str(item.memory),
        "gpu_count": str(item.gpu_count),
        "gpu_name": _dump_optional(item.gpu_name),
        "gpu_memory": _dump_optional(item.gpu_memory),
        "spot": str(item.spot),
        "disk_size": _dump_optional(item.disk_size),
        "gpu_vendor": _dump_optional(item.gpu_vendor.value if item.gpu_vendor else None),
        "flags": " ".join(item.flags),
        "cpu_arch": _dump_optional(item.cpu_arch.value if item.cpu_arch else None),
        "provider_data": json.dumps(item.provider_data),
    }


def item_from_row(row: Mapping[str, str], *, provider: str) -> CatalogItem:
    gpu_name = load_optional(row.get("gpu_name"))
    gpu_vendor = load_optional(row.get("gpu_vendor"))
    # Catalogs published before the `gpu_vendor` column existed encode TPUs by prefixing
    # the accelerator name. `cpu_arch` predates its column too, and is filled in by
    # `CatalogItem.__post_init__`. Both heuristics are required as long as we support
    # historical catalogs.
    if gpu_name and gpu_name.startswith("tpu-"):
        gpu_name = gpu_name[4:]
        if gpu_vendor is None:
            gpu_vendor = AcceleratorVendor.GOOGLE.value
    cpu_arch = load_optional(row.get("cpu_arch"))
    return CatalogItem(
        provider=provider,
        instance_name=load_required(row.get("instance_name")),
        location=load_required(row.get("location")),
        price=load_required(row.get("price"), loader=float),
        cpu=load_required(row.get("cpu"), loader=int),
        memory=load_required(row.get("memory"), loader=float),
        gpu_count=load_required(row.get("gpu_count"), loader=int),
        gpu_name=gpu_name,
        gpu_memory=load_optional(row.get("gpu_memory"), loader=float),
        spot=load_required(row.get("spot"), loader=bool_loader),
        disk_size=load_optional(row.get("disk_size"), loader=float),
        gpu_vendor=AcceleratorVendor.cast(gpu_vendor) if gpu_vendor else None,
        flags=(row.get("flags") or "").split(),
        cpu_arch=CPUArchitecture.cast(cpu_arch) if cpu_arch else None,
        provider_data=json.loads(row.get("provider_data") or "{}"),
    )


def dump(items: Iterable[CatalogItem], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_V2_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(item_to_row(item))


def load(f: IO[str], *, provider: str) -> Iterator[CatalogItem]:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            yield item_from_row(row, provider=provider)
        except ValueError:
            logger.exception(
                "Skipping malformed row in %s catalog at line %s", provider, reader.line_num
            )


def convert_catalog_v2_to_v1(path_v2: str, path_v1: str) -> None:
    with open(path_v2) as f_v2, open(path_v1, "w") as f_v1:
        reader = csv.DictReader(f_v2)
        writer = csv.DictWriter(f_v1, fieldnames=CATALOG_V1_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            if not row.get("flags"):
                writer.writerow(row)


def _dump_optional(value: str | float | None) -> str:
    return "" if value is None else str(value)
