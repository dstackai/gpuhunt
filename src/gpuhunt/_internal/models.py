import enum
from dataclasses import asdict, dataclass, fields
from typing import (
    ClassVar,
    Optional,
    Union,
)

from gpuhunt._internal.utils import empty_as_none


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


class AcceleratorVendor(enum.Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    GOOGLE = "google"

    @classmethod
    def cast(cls, value: Union["AcceleratorVendor", str]) -> "AcceleratorVendor":
        if isinstance(value, AcceleratorVendor):
            return value
        return cls(value.lower())


@dataclass
class RawCatalogItem:
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

    def __post_init__(self) -> None:
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

    @staticmethod
    def from_dict(v: dict) -> "RawCatalogItem":
        return RawCatalogItem(
            instance_name=empty_as_none(v.get("instance_name")),
            location=empty_as_none(v.get("location")),
            price=empty_as_none(v.get("price"), loader=float),
            cpu=empty_as_none(v.get("cpu"), loader=int),
            memory=empty_as_none(v.get("memory"), loader=float),
            gpu_vendor=empty_as_none(v.get("gpu_vendor")),
            gpu_count=empty_as_none(v.get("gpu_count"), loader=int),
            gpu_name=empty_as_none(v.get("gpu_name")),
            gpu_memory=empty_as_none(v.get("gpu_memory"), loader=float),
            spot=empty_as_none(v.get("spot"), loader=bool_loader),
            disk_size=empty_as_none(v.get("disk_size"), loader=float),
        )

    def dict(self) -> dict[str, Union[str, int, float, bool, None]]:
        return asdict(self)


@dataclass
class CatalogItem:
    """
    Attributes:
        instance_name: name of the instance
        location: region or zone
        price: $ per hour
        cpu: number of CPUs
        memory: amount of RAM in GB
        gpu_count: number of GPUs
        gpu_name: name of the GPU
        gpu_memory: amount of GPU VRAM in GB for each GPU
        spot: whether the instance is a spot instance
        provider: name of the provider
        disk_size: size of disk in GB
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

    def __post_init__(self) -> None:
        gpu_vendor = self.gpu_vendor
        if gpu_vendor is None:
            # This heuristic is only required until we update all providers to always set
            # the vendor.
            if not self.gpu_count:
                # None or 0
                return
            # GCPProvider already sets gpu_vendor, and all other providers only support Nvidia
            self.gpu_vendor = AcceleratorVendor.NVIDIA
        else:
            # This cast to the enum is always required since RawCatalogItem.gpu_vendor
            # is a string field (for (de)serialization purposes).
            self.gpu_vendor = AcceleratorVendor.cast(gpu_vendor)

    @staticmethod
    def from_dict(v: dict, *, provider: Optional[str] = None) -> "CatalogItem":
        return CatalogItem(provider=provider, **asdict(RawCatalogItem.from_dict(v)))


@dataclass
class QueryFilter:
    """
    Attributes:
        provider: name of the provider to filter by. If not specified, all providers will be used
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
    """

    provider: Optional[list[str]] = None  # strings can have mixed case
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


@dataclass
class TPUInfo(AcceleratorInfo):
    vendor = AcceleratorVendor.GOOGLE
