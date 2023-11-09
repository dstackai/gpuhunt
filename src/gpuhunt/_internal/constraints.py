import copy
from typing import Optional, Tuple, TypeVar, Union

from gpuhunt._internal.models import CatalogItem, GPUInfo, QueryFilter


def fill_missing(q: QueryFilter, *, memory_per_core: int = 6) -> QueryFilter:
    q = copy.deepcopy(q)

    # if there is some information about gpu
    min_total_gpu_memory = None
    if any(
        value is not None
        for value in (
            q.gpu_name,
            q.min_gpu_count,
            q.min_gpu_memory,
            q.min_total_gpu_memory,
            q.min_compute_capability,
        )
    ):
        if q.min_total_gpu_memory is not None:
            min_total_gpu_memory = q.min_total_gpu_memory
        else:
            min_gpu_count = 1 if q.min_gpu_count is None else q.min_gpu_count
            min_gpu_memory = []
            if q.min_gpu_memory is not None:
                min_gpu_memory.append(q.min_gpu_memory)
            gpus = KNOWN_GPUS
            if q.min_compute_capability is not None:  # filter gpus by compute capability
                gpus = [i for i in gpus if i.compute_capability >= q.min_compute_capability]
            if q.gpu_name is not None:  # filter gpus by name
                gpus = [i for i in gpus if i.name.lower() in q.gpu_name]
            min_gpu_memory.append(
                min((i.memory for i in gpus), default=min(i.memory for i in KNOWN_GPUS))
            )
            min_total_gpu_memory = max(min_gpu_memory) * min_gpu_count

    if min_total_gpu_memory is not None:
        if q.min_memory is None:  # gpu memory to memory
            q.min_memory = 2 * min_total_gpu_memory
        if q.min_disk_size is None:  # gpu memory to disk
            q.min_disk_size = 30 + min_total_gpu_memory

    if q.min_memory is not None:
        if q.min_cpu is None:  # memory to cpu
            q.min_cpu = (q.min_memory + memory_per_core - 1) // memory_per_core

    if q.min_cpu is not None:
        if q.min_memory is None:  # cpu to memory
            q.min_memory = memory_per_core * q.min_cpu

    return q


Number = TypeVar("Number", bound=Union[int, float])


def optimize(
    available: Number, min_limit: Number, max_limit: Optional[Number]
) -> Optional[Number]:
    if is_above(available, max_limit):
        available = max_limit
    if is_below(available, min_limit):
        return None
    return min_limit


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
    if q.provider is not None and i.provider.lower() not in q.provider:
        return False
    if not is_between(i.cpu, q.min_cpu, q.max_cpu):
        return False
    if not is_between(i.memory, q.min_memory, q.max_memory):
        return False
    if not is_between(i.gpu_count, q.min_gpu_count, q.max_gpu_count):
        return False
    if q.gpu_name is not None:
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
    # TODO(egor-s): add disk_size to CatalogItem
    # if not is_between(i.disk_size, q.min_disk_size, q.max_disk_size):
    #     return False
    if not is_between(i.price, q.min_price, q.max_price):
        return False
    if q.spot is not None and i.spot != q.spot:
        return False
    return True


def get_compute_capability(gpu_name: str) -> Optional[Tuple[int, int]]:
    for gpu in KNOWN_GPUS:
        if gpu.name.lower() == gpu_name.lower():
            return gpu.compute_capability
    return None


KNOWN_GPUS = [
    GPUInfo(name="A10", memory=24, compute_capability=(8, 6)),
    GPUInfo(name="A100", memory=40, compute_capability=(8, 0)),
    GPUInfo(name="A100", memory=80, compute_capability=(8, 0)),
    GPUInfo(name="A10G", memory=24, compute_capability=(8, 6)),
    GPUInfo(name="A4000", memory=16, compute_capability=(8, 6)),
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
