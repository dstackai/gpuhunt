from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Union

from gpuhunt._internal.utils import is_between, empty_as_none


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
            spot=empty_as_none(v.get("spot"), loader=lambda x: x.lower() == "true"),
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
        gpu_name: case-sensitive name of the GPU to filter by. If not specified, all GPUs will be used
        min_gpu_memory: minimum amount of GPU VRAM in GB for each GPU
        max_gpu_memory: maximum amount of GPU VRAM in GB for each GPU
        min_total_gpu_memory: minimum amount of GPU VRAM in GB for all GPUs combined
        max_total_gpu_memory: maximum amount of GPU VRAM in GB for all GPUs combined
        min_disk_size: *currently not in use*
        max_disk_size: *currently not in use*
        min_price: minimum price per hour in USD
        max_price: maximum price per hour in USD
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
    spot: Optional[bool] = None

    def matches(self, item: CatalogItem) -> bool:
        """
        Check if the catalog item matches the filters

        Args:
            item: catalog item

        Returns:
            whether the catalog item matches the filters
        """
        if self.provider is not None and item.provider not in self.provider:
            return False
        if not is_between(item.cpu, self.min_cpu, self.max_cpu):
            return False
        if not is_between(item.memory, self.min_memory, self.max_memory):
            return False
        if not is_between(item.gpu_count, self.min_gpu_count, self.max_gpu_count):
            return False
        if self.gpu_name is not None and item.gpu_name not in self.gpu_name:
            return False
        if not is_between(item.gpu_memory if item.gpu_count > 0 else 0, self.min_gpu_memory, self.max_gpu_memory):
            return False
        if not is_between(
                (item.gpu_count * item.gpu_memory) if item.gpu_count > 0 else 0, self.min_total_gpu_memory, self.max_total_gpu_memory
        ):
            return False
        # TODO(egor-s): add disk_size to CatalogItem
        # if not is_between(item.disk_size, self.min_disk_size, self.max_disk_size):
        #     return False
        if not is_between(item.price, self.min_price, self.max_price):
            return False
        if self.spot is not None and item.spot != self.spot:
            return False
        return True
