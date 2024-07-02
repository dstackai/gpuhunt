from typing import Optional, Tuple, TypeVar, Union

from gpuhunt._internal.models import CatalogItem, GPUInfo, QueryFilter
from gpuhunt._internal.utils import _is_tpu

Comparable = TypeVar("Comparable", bound=Union[int, float, Tuple[int, int]])


def is_between(value: Comparable, left: Optional[Comparable], right: Optional[Comparable]) -> bool:
    if is_below(value, left) or is_above(value, right):
        return False
    return True


def is_below(value: Comparable, limit: Optional[Comparable]) -> bool:
    if limit is not None and value < limit:
        return True
    return False


def is_above(value: Comparable, limit: Optional[Comparable]) -> bool:
    if limit is not None and value > limit:
        return True
    return False


def matches(i: CatalogItem, q: QueryFilter) -> bool:
    """
    Check if the catalog item matches the filters

    Args:
        i: catalog item
        q: filters

    Returns:
        whether the catalog item matches the filters
    """
    # Common checks
    if q.provider is not None and i.provider.lower() not in q.provider:
        return False
    if not is_between(i.price, q.min_price, q.max_price):
        return False
    if q.spot is not None and i.spot != q.spot:
        return False

    # TPU specific checks
    if i.gpu_name and _is_tpu(i.gpu_name.lower()):
        if q.gpu_name is not None:
            if i.gpu_name is None:
                return False
            if i.gpu_name.lower() not in q.gpu_name:
                return False
        return True

    # GPU & CPU checks
    if not is_between(i.cpu, q.min_cpu, q.max_cpu):
        return False
    if not is_between(i.memory, q.min_memory, q.max_memory):
        return False
    if not is_between(i.gpu_count, q.min_gpu_count, q.max_gpu_count):
        return False
    if q.gpu_name is not None:
        if i.gpu_name is None:
            return False
        if i.gpu_name.lower() not in q.gpu_name:
            return False
    if q.min_compute_capability is not None or q.max_compute_capability is not None:
        cc = get_compute_capability(i.gpu_name)
        if not cc or not is_between(cc, q.min_compute_capability, q.max_compute_capability):
            return False
    if not is_between(i.gpu_memory if i.gpu_count > 0 else 0, q.min_gpu_memory, q.max_gpu_memory):
        return False
    if not is_between(
        (i.gpu_count * i.gpu_memory) if i.gpu_count > 0 else 0,
        q.min_total_gpu_memory,
        q.max_total_gpu_memory,
    ):
        return False
    if i.disk_size is not None:
        if not is_between(i.disk_size, q.min_disk_size, q.max_disk_size):
            return False
    return True


def get_compute_capability(gpu_name: str) -> Optional[Tuple[int, int]]:
    for gpu in KNOWN_GPUS:
        if gpu.name.lower() == gpu_name.lower():
            return gpu.compute_capability
    return None


KNOWN_GPUS = [
    GPUInfo(name="A10", memory=24, compute_capability=(8, 6)),
    GPUInfo(name="A40", memory=48, compute_capability=(8, 6)),
    GPUInfo(name="A100", memory=40, compute_capability=(8, 0)),
    GPUInfo(name="A100", memory=80, compute_capability=(8, 0)),
    GPUInfo(name="A10G", memory=24, compute_capability=(8, 6)),
    GPUInfo(name="A4000", memory=16, compute_capability=(8, 6)),
    GPUInfo(name="A4500", memory=20, compute_capability=(8, 6)),
    GPUInfo(name="A5000", memory=24, compute_capability=(8, 6)),
    GPUInfo(name="A6000", memory=48, compute_capability=(8, 6)),
    GPUInfo(name="H100", memory=80, compute_capability=(9, 0)),
    GPUInfo(name="L4", memory=24, compute_capability=(8, 9)),
    GPUInfo(name="L40", memory=48, compute_capability=(8, 9)),
    GPUInfo(name="P100", memory=16, compute_capability=(6, 0)),
    GPUInfo(name="RTX3060", memory=8, compute_capability=(8, 6)),
    GPUInfo(name="RTX3060", memory=12, compute_capability=(8, 6)),
    GPUInfo(name="RTX3060Ti", memory=8, compute_capability=(8, 6)),
    GPUInfo(name="RTX3070Ti", memory=8, compute_capability=(8, 6)),
    GPUInfo(name="RTX3080", memory=10, compute_capability=(8, 6)),
    GPUInfo(name="RTX3080Ti", memory=12, compute_capability=(8, 6)),
    GPUInfo(name="RTX3090", memory=24, compute_capability=(8, 6)),
    GPUInfo(name="RTX4090", memory=24, compute_capability=(8, 9)),
    GPUInfo(name="RTX6000", memory=24, compute_capability=(7, 5)),
    GPUInfo(name="T4", memory=16, compute_capability=(7, 5)),
    GPUInfo(name="V100", memory=16, compute_capability=(7, 0)),
    GPUInfo(name="V100", memory=32, compute_capability=(7, 0)),
]
