import csv
import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import IO

from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, CPUArchitecture
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
        "cpu_arch": item.cpu_arch.value,
        "provider_data": json.dumps(item.provider_data),
    }


def item_from_row(row: Mapping[str, str], *, provider: str) -> CatalogItem:
    gpu_count = load_required(row.get("gpu_count"), loader=int)
    gpu_name = load_optional(row.get("gpu_name"))
    raw_gpu_vendor = load_optional(row.get("gpu_vendor"))
    gpu_vendor = AcceleratorVendor.cast(raw_gpu_vendor) if raw_gpu_vendor else None
    # Catalogs published before the `gpu_vendor` column existed encode TPUs by prefixing
    # the accelerator name, and imply Nvidia otherwise. Required as long as we support
    # historical catalogs.
    if gpu_name and gpu_name.startswith("tpu-"):
        gpu_name = gpu_name[4:]
        if gpu_vendor is None:
            gpu_vendor = AcceleratorVendor.GOOGLE
    elif gpu_vendor is None and gpu_count:
        gpu_vendor = AcceleratorVendor.NVIDIA
    # `cpu_arch` predates its column too, and x86 is what those catalogs contain.
    raw_cpu_arch = load_optional(row.get("cpu_arch"))
    cpu_arch = CPUArchitecture.cast(raw_cpu_arch) if raw_cpu_arch else CPUArchitecture.X86
    return CatalogItem(
        provider=provider,
        instance_name=load_required(row.get("instance_name")),
        location=load_required(row.get("location")),
        price=load_required(row.get("price"), loader=float),
        cpu=load_required(row.get("cpu"), loader=int),
        memory=load_required(row.get("memory"), loader=float),
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=load_optional(row.get("gpu_memory"), loader=float),
        spot=load_required(row.get("spot"), loader=_load_bool),
        disk_size=load_optional(row.get("disk_size"), loader=float),
        gpu_vendor=gpu_vendor,
        flags=(row.get("flags") or "").split(),
        cpu_arch=cpu_arch,
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


def _load_bool(value: str) -> bool:
    return value.lower() == "true"
