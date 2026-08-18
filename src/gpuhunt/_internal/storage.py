import csv
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import IO, TypeVar, overload

from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, CPUArchitecture

R = TypeVar("R")

logger = logging.getLogger(__name__)

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
    gpu_count = _load_required(row, "gpu_count", int)
    gpu_name = _load_optional(row, "gpu_name")
    raw_gpu_vendor = _load_optional(row, "gpu_vendor")
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
    raw_cpu_arch = _load_optional(row, "cpu_arch")
    cpu_arch = CPUArchitecture.cast(raw_cpu_arch) if raw_cpu_arch else CPUArchitecture.X86
    return CatalogItem(
        provider=provider,
        instance_name=_load_required(row, "instance_name"),
        location=_load_required(row, "location"),
        price=_load_required(row, "price", float),
        cpu=_load_required(row, "cpu", int),
        memory=_load_required(row, "memory", float),
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=_load_optional(row, "gpu_memory", float),
        spot=_load_required(row, "spot", _load_bool),
        disk_size=_load_optional(row, "disk_size", float),
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
        except ValueError as e:
            logger.warning(
                "Skipping malformed row in %s catalog at line %s: %s", provider, reader.line_num, e
            )


def _dump_optional(value: str | float | None) -> str:
    return "" if value is None else str(value)


@overload
def _load_required(row: Mapping[str, str], field: str, loader: Callable[[str], R]) -> R: ...


@overload
def _load_required(row: Mapping[str, str], field: str, loader: None = None) -> str: ...


def _load_required(
    row: Mapping[str, str], field: str, loader: Callable[[str], R] | None = None
) -> str | R:
    value = row.get(field)
    if value is None:
        raise ValueError(f"Required field {field!r} is missing")
    if value == "":
        raise ValueError(f"Required field {field!r} is empty")
    return _apply_loader(field, value, loader)


@overload
def _load_optional(row: Mapping[str, str], field: str, loader: Callable[[str], R]) -> R | None: ...


@overload
def _load_optional(row: Mapping[str, str], field: str, loader: None = None) -> str | None: ...


def _load_optional(
    row: Mapping[str, str], field: str, loader: Callable[[str], R] | None = None
) -> str | R | None:
    value = row.get(field)
    if not value:
        return None
    return _apply_loader(field, value, loader)


def _apply_loader(field: str, value: str, loader: Callable[[str], R] | None) -> str | R:
    if loader is None:
        return value
    try:
        return loader(value)
    except ValueError as e:
        raise ValueError(f"Cannot parse field {field!r}: {e}") from e


def _load_bool(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise ValueError(f"Not a boolean: {value!r}")
    return value.lower() == "true"
