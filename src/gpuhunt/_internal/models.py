import enum
from collections.abc import Container
from dataclasses import dataclass, field, fields
from typing import (
    ClassVar,
    Union,
)

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


class AMDArchitecture(enum.Enum):
    CDNA = "CDNA"
    CDNA2 = "CDNA2"
    CDNA3 = "CDNA3"
    CDNA4 = "CDNA4"

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

    provider: str
    instance_name: str
    location: str
    price: float
    cpu: int
    memory: float
    gpu_count: int
    gpu_name: str | None
    gpu_memory: float | None
    spot: bool
    disk_size: float | None
    gpu_vendor: AcceleratorVendor | None = None
    flags: list[str] = field(default_factory=list)
    cpu_arch: CPUArchitecture = CPUArchitecture.X86
    provider_data: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gpu_count and self.gpu_vendor is None:
            raise ValueError(f"gpu_vendor is required when gpu_count is non-zero: {self}")


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

    provider: list[str] | None = None  # strings can have mixed case
    cpu_arch: CPUArchitecture | None = None
    min_cpu: int | None = None
    max_cpu: int | None = None
    min_memory: float | None = None
    max_memory: float | None = None
    min_gpu_count: int | None = None
    max_gpu_count: int | None = None
    gpu_vendor: AcceleratorVendor | None = None
    gpu_name: list[str] | None = None  # strings can have mixed case
    min_gpu_memory: float | None = None
    max_gpu_memory: float | None = None
    min_total_gpu_memory: float | None = None
    max_total_gpu_memory: float | None = None
    min_disk_size: int | None = None
    max_disk_size: int | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_compute_capability: tuple[int, int] | None = None
    max_compute_capability: tuple[int, int] | None = None
    spot: bool | None = None
    allowed_flags: Container[str] | None = None

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
