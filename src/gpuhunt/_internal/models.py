from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple, Union

from gpuhunt._internal.utils import empty_as_none


def bool_loader(x: Union[bool, str]) -> bool:
    if isinstance(x, bool):
        return x
    return x.lower() == "true"


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

    @staticmethod
    def from_dict(v: dict) -> "RawCatalogItem":
        return RawCatalogItem(
            instance_name=empty_as_none(v.get("instance_name")),
            location=empty_as_none(v.get("location")),
            price=empty_as_none(v.get("price"), loader=float),
            cpu=empty_as_none(v.get("cpu"), loader=int),
            memory=empty_as_none(v.get("memory"), loader=float),
            gpu_count=empty_as_none(v.get("gpu_count"), loader=int),
            gpu_name=empty_as_none(v.get("gpu_name")),
            gpu_memory=empty_as_none(v.get("gpu_memory"), loader=float),
            spot=empty_as_none(v.get("spot"), loader=bool_loader),
            disk_size=empty_as_none(v.get("disk_size"), loader=float),
        )

    def dict(self) -> Dict[str, Union[str, int, float, bool, None]]:
        return asdict(self)


@dataclass
class CatalogItem(RawCatalogItem):
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
    provider: str
    disk_size: Optional[float]

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

    provider: Optional[List[str]] = None
    min_cpu: Optional[int] = None
    max_cpu: Optional[int] = None
    min_memory: Optional[float] = None
    max_memory: Optional[float] = None
    min_gpu_count: Optional[int] = None
    max_gpu_count: Optional[int] = None
    gpu_name: Optional[List[str]] = None
    min_gpu_memory: Optional[float] = None
    max_gpu_memory: Optional[float] = None
    min_total_gpu_memory: Optional[float] = None
    max_total_gpu_memory: Optional[float] = None
    min_disk_size: Optional[int] = None
    max_disk_size: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_compute_capability: Optional[Tuple[int, int]] = None
    max_compute_capability: Optional[Tuple[int, int]] = None
    spot: Optional[bool] = None

    def __post_init__(self):
        if self.provider is not None:
            self.provider = [i.lower() for i in self.provider]
        if self.gpu_name is not None:
            self.gpu_name = [i.lower() for i in self.gpu_name]

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
class GPUInfo:
    name: str
    memory: int
    compute_capability: Tuple[int, int]
