import enum
import json
from collections.abc import Container
from dataclasses import asdict, dataclass, field, fields
from typing import (
    ClassVar,
    Optional,
    Union,
)

from gpuhunt._internal.utils import empty_as_none

JSONType = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JSONType"],
    "JSONObject",
]
JSONObject = dict[str, JSONType]


def bool_loader(x: Union[bool, str]) -> bool:
    if isinstance(x, bool):
        return x
    return x.lower() == "true"


class AMDArchitecture(enum.Enum):
    CDNA = "CDNA"
    CDNA2 = "CDNA2"
    CDNA3 = "CDNA3"

    @classmethod
    def cast(cls, value: Union["AMDArchitecture", str]) -> "AMDArchitecture":
        if isinstance(value, AMDArchitecture):
            return value
        return cls(value.upper())


class AcceleratorVendor(str, enum.Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    GOOGLE = "google"
    INTEL = "intel"
    TENSTORRENT = "tenstorrent"

    @classmethod
    def cast(cls, value: Union["AcceleratorVendor", str]) -> "AcceleratorVendor":
        if isinstance(value, AcceleratorVendor):
            return value
        return cls(value.lower())


class CPUArchitecture(str, enum.Enum):
    X86 = "x86"  # x86-64 extension support implied
    ARM = "arm"  # AArch64 (ARM64) execution state support implied

    @classmethod
    def cast(cls, value: Union["CPUArchitecture", str]) -> "CPUArchitecture":
        if isinstance(value, CPUArchitecture):
            return value
        return cls(value.lower())


@dataclass
class RawCatalogItem:
    """
    An item stored in the catalog.
    See `CatalogItem` for field descriptions.
    """

    instance_name: Optional[str]
    location: Optional[str]
    price: Optional[float]
    cpu: Optional[int]
    memory: Optional[float]
    gpu_count: Optional[int]
    gpu_name: Optional[str]
    gpu_memory: Optional[float]
    spot: Optional[bool]
    disk_size: Optional[float]
    gpu_vendor: Optional[str] = None
    flags: list[str] = field(default_factory=list)
    cpu_arch: Optional[str] = None
    provider_data: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._process_gpu_vendor()
        self._process_cpu_arch()

    def _process_gpu_vendor(self) -> None:
        # This heuristic will be required indefinitely since we support historical catalogs.
        is_tpu = False
        gpu_name = self.gpu_name
        if gpu_name and gpu_name.startswith("tpu-"):
            is_tpu = True
            self.gpu_name = gpu_name[4:]
        gpu_vendor = self.gpu_vendor
        if gpu_vendor is None:
            if not self.gpu_count:
                # None or 0
                return
            if is_tpu:
                self.gpu_vendor = AcceleratorVendor.GOOGLE.value
            else:
                self.gpu_vendor = AcceleratorVendor.NVIDIA.value
        elif isinstance(gpu_vendor, AcceleratorVendor):
            self.gpu_vendor = gpu_vendor.value

    def _process_cpu_arch(self) -> None:
        # This heuristic will be required indefinitely since we support historical catalogs.
        cpu_arch = self.cpu_arch
        if cpu_arch is None:
            self.cpu_arch = CPUArchitecture.X86.value
        elif isinstance(cpu_arch, CPUArchitecture):
            self.cpu_arch = cpu_arch.value

    @staticmethod
    def from_dict(v: dict) -> "RawCatalogItem":
        return RawCatalogItem(
            instance_name=empty_as_none(v.get("instance_name")),
            location=empty_as_none(v.get("location")),
            price=empty_as_none(v.get("price"), loader=float),
            cpu_arch=empty_as_none(v.get("cpu_arch")),
            cpu=empty_as_none(v.get("cpu"), loader=int),
            memory=empty_as_none(v.get("memory"), loader=float),
            gpu_vendor=empty_as_none(v.get("gpu_vendor")),
            gpu_count=empty_as_none(v.get("gpu_count"), loader=int),
            gpu_name=empty_as_none(v.get("gpu_name")),
            gpu_memory=empty_as_none(v.get("gpu_memory"), loader=float),
            spot=empty_as_none(v.get("spot"), loader=bool_loader),
            disk_size=empty_as_none(v.get("disk_size"), loader=float),
            flags=v.get("flags", "").split(),
            provider_data=json.loads(v.get("provider_data", "{}")),
        )

    def dict(self) -> dict[str, Union[str, int, float, bool, None]]:
        return {
            **asdict(self),
            "flags": " ".join(self.flags),
            "provider_data": json.dumps(self.provider_data),
        }


@dataclass
class CatalogItem:
    """
    An item returned by `Catalog.query`.
    Attributes:
        instance_name: name of the instance
        location: region or zone
        price: $ per hour
        cpu_arch: CPU instruction set architecture
        cpu: number of CPUs
        memory: amount of RAM in GB
        gpu_vendor: GPU/accelerator vendor
        gpu_count: number of GPUs
        gpu_name: name of the GPU
        gpu_memory: amount of GPU VRAM in GB for each GPU
        spot: whether the instance is a spot instance
        provider: name of the provider
        disk_size: size of disk in GB
        flags: list of flags. If a catalog item breaks existing dstack versions,
            add a flag to hide the item from those versions. Newer dstack versions
            will have to request this flag explicitly to get the catalog item.
            If you are adding a new provider, leave the flags empty.
            Flag names should be in kebab-case.
        provider_data: dict with provider-specific properties.
            Prefer defining a TypedDict within provider implementation.
    """

    instance_name: str
    location: str
    price: float
    cpu: int
    memory: float
    gpu_count: int
    gpu_name: Optional[str]
    gpu_memory: Optional[float]
    spot: bool
    disk_size: Optional[float]
    provider: str
    gpu_vendor: Optional[AcceleratorVendor] = None
    flags: list[str] = field(default_factory=list)
    cpu_arch: Optional[CPUArchitecture] = None
    provider_data: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._process_gpu_vendor()
        self._process_cpu_arch()

    def _process_gpu_vendor(self) -> None:
        # This heuristic is only required until we update all providers to always set the vendor.
        gpu_vendor = self.gpu_vendor
        if gpu_vendor is None:
            if not self.gpu_count:
                # None or 0
                return
            # GCPProvider already sets gpu_vendor, and all other providers only support Nvidia
            self.gpu_vendor = AcceleratorVendor.NVIDIA
        else:
            # This cast to the enum is always required since RawCatalogItem.gpu_vendor
            # is a string field (for (de)serialization purposes).
            self.gpu_vendor = AcceleratorVendor.cast(gpu_vendor)

    def _process_cpu_arch(self) -> None:
        # This heuristic is only required until we update all providers to always set the arch.
        cpu_arch = self.cpu_arch
        if cpu_arch is None:
            self.cpu_arch = CPUArchitecture.X86
        else:
            self.cpu_arch = CPUArchitecture.cast(cpu_arch)

    @staticmethod
    def from_dict(v: dict, *, provider: Optional[str] = None) -> "CatalogItem":
        return CatalogItem(provider=provider, **asdict(RawCatalogItem.from_dict(v)))


@dataclass
class QueryFilter:
    """
    Attributes:
        provider: name of the provider to filter by. If not specified, all providers will be used
        cpu_arch: CPU architecture. If not specified, all architectures will be used
        min_cpu: minimum number of CPUs
        max_cpu: maximum number of CPUs
        min_memory: minimum amount of RAM in GB
        max_memory: maximum amount of RAM in GB
        min_gpu_count: minimum number of GPUs
        max_gpu_count: maximum number of GPUs
        gpu_vendor: accelerator vendor to filter by. If not specified, all vendors will be used
        gpu_name: name of the GPU to filter by. If not specified, all GPUs will be used
        min_gpu_memory: minimum amount of GPU VRAM in GB for each GPU
        max_gpu_memory: maximum amount of GPU VRAM in GB for each GPU
        min_total_gpu_memory: minimum amount of GPU VRAM in GB for all GPUs combined
        max_total_gpu_memory: maximum amount of GPU VRAM in GB for all GPUs combined
        min_disk_size: minimum disk size in GB
        max_disk_size: maximum disk size in GB
        min_price: minimum price per hour in USD
        max_price: maximum price per hour in USD
        min_compute_capability: minimum compute capability of the GPU
        max_compute_capability: maximum compute capability of the GPU
        spot: if `False`, only ondemand offers will be returned. If `True`, only spot offers will be returned
        allowed_flags: only offers with all flags allowed will be returned. `None` allows all flags
    """

    provider: Optional[list[str]] = None  # strings can have mixed case
    cpu_arch: Optional[CPUArchitecture] = None
    min_cpu: Optional[int] = None
    max_cpu: Optional[int] = None
    min_memory: Optional[float] = None
    max_memory: Optional[float] = None
    min_gpu_count: Optional[int] = None
    max_gpu_count: Optional[int] = None
    gpu_vendor: Optional[AcceleratorVendor] = None
    gpu_name: Optional[list[str]] = None  # strings can have mixed case
    min_gpu_memory: Optional[float] = None
    max_gpu_memory: Optional[float] = None
    min_total_gpu_memory: Optional[float] = None
    max_total_gpu_memory: Optional[float] = None
    min_disk_size: Optional[int] = None
    max_disk_size: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_compute_capability: Optional[tuple[int, int]] = None
    max_compute_capability: Optional[tuple[int, int]] = None
    spot: Optional[bool] = None
    allowed_flags: Optional[Container[str]] = None

    def __repr__(self) -> str:
        """
        >>> QueryFilter()
        QueryFilter()
        >>> QueryFilter(min_cpu=4)
        QueryFilter(min_cpu=4)
        >>> QueryFilter(max_price=1.2, min_cpu=4)
        QueryFilter(min_cpu=4, max_price=1.2)
        """
        kv = ", ".join(
            f"{f.name}={value}"
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        )
        return f"QueryFilter({kv})"


@dataclass
class AcceleratorInfo:
    vendor: ClassVar[AcceleratorVendor]
    name: str
    memory: int


@dataclass
class NvidiaGPUInfo(AcceleratorInfo):
    vendor = AcceleratorVendor.NVIDIA
    compute_capability: tuple[int, int]


@dataclass
class AMDGPUInfo(AcceleratorInfo):
    vendor = AcceleratorVendor.AMD
    architecture: AMDArchitecture
    device_ids: tuple[int, ...]


@dataclass
class TPUInfo(AcceleratorInfo):
    vendor = AcceleratorVendor.GOOGLE


@dataclass
class IntelAcceleratorInfo(AcceleratorInfo):
    vendor = AcceleratorVendor.INTEL


@dataclass
class TenstorrentAcceleratorInfo(AcceleratorInfo):
    vendor = AcceleratorVendor.TENSTORRENT
