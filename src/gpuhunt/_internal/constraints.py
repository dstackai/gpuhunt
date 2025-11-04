import re
from collections.abc import Container, Iterable
from typing import Optional, TypeVar, Union

from gpuhunt._internal.models import (
    AcceleratorInfo,
    AcceleratorVendor,
    AMDArchitecture,
    AMDGPUInfo,
    CatalogItem,
    IntelAcceleratorInfo,
    NvidiaGPUInfo,
    QueryFilter,
    TenstorrentAcceleratorInfo,
    TPUInfo,
)

# v5litepod = v5e
_TPU_VERSIONS = ["v2", "v3", "v4", "v5p", "v5litepod", "v6e"]


Comparable = TypeVar("Comparable", bound=Union[int, float, tuple[int, int]])


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
    if q.provider is not None and i.provider.lower() not in map(str.lower, q.provider):
        return False
    if not is_between(i.price, q.min_price, q.max_price):
        return False
    if q.spot is not None and i.spot != q.spot:
        return False
    if q.cpu_arch and q.cpu_arch != i.cpu_arch:
        return False
    if not is_between(i.cpu, q.min_cpu, q.max_cpu):
        return False
    if not is_between(i.memory, q.min_memory, q.max_memory):
        return False
    if not (q.min_gpu_count == 0 and i.gpu_count == 0):
        # GPU filters should not be applied to non-gpu offers if `q.min_gpu_count == 0`.
        if q.gpu_vendor and q.gpu_vendor != i.gpu_vendor:
            return False
        if not is_between(i.gpu_count, q.min_gpu_count, q.max_gpu_count):
            return False
        if q.gpu_name is not None:
            if i.gpu_name is None:
                return False
            if i.gpu_name.lower() not in map(str.lower, q.gpu_name):
                return False
        if q.min_compute_capability is not None or q.max_compute_capability is not None:
            if i.gpu_vendor != AcceleratorVendor.NVIDIA:
                return False
            if not i.gpu_name:
                return False
            cc = get_compute_capability(i.gpu_name)
            if not cc or not is_between(cc, q.min_compute_capability, q.max_compute_capability):
                return False
        if not is_between(
            i.gpu_memory if i.gpu_count > 0 else 0, q.min_gpu_memory, q.max_gpu_memory
        ):
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
    if q.allowed_flags is not None:
        if any(flag not in q.allowed_flags for flag in i.flags):
            return False
    return True


def find_accelerators(
    names: Optional[Iterable[str]] = None, vendors: Optional[Container[AcceleratorVendor]] = None
) -> list[AcceleratorInfo]:
    if names is not None:
        names = {n.lower() for n in names}
    result = []
    for accelerator in KNOWN_ACCELERATORS:
        if (names is None or accelerator.name.lower() in names) and (
            vendors is None or accelerator.vendor in vendors
        ):
            result.append(accelerator)
    return result


def get_compute_capability(gpu_name: str) -> Optional[tuple[int, int]]:
    if accelerators := find_accelerators(names=[gpu_name], vendors=AcceleratorVendor.NVIDIA):
        assert isinstance(accelerators[0], NvidiaGPUInfo)
        return accelerators[0].compute_capability
    return None


def get_gpu_vendor(gpu_name: Optional[str]) -> Optional[AcceleratorVendor]:
    if gpu_name is None:
        return None
    if accelerators := find_accelerators(names=[gpu_name]):
        return accelerators[0].vendor
    return None


def correct_gpu_memory_gib(gpu_name: str, memory_mib: float) -> int:
    """
    Round to whole number of gibibytes and attempt correcting the reported GPU
    memory size if the actual memory size for that GPU is known and the
    difference between the reported and the known memory is within a heuristic
    threshold.

    This is useful for cases when nvidia-smi or cloud providers report the GPU
    memory imprecisely.
    """

    memory_gib = memory_mib / 1024
    known_memories_gib = {gpu.memory for gpu in find_accelerators(names=[gpu_name])}
    if known_memories_gib:
        closest_known_memory_gib = min(known_memories_gib, key=lambda x: abs(x - memory_gib))
        difference_gib = abs(closest_known_memory_gib - memory_gib)
        if difference_gib / closest_known_memory_gib < 0.07:
            return closest_known_memory_gib
    return round(memory_gib)


def is_nvidia_superchip(gpu_name: str) -> bool:
    """
    Check if the given NVIDIA GPU is actually a so-called "superchip" combining GPU with ARM CPU,
    such as:
    * GH200 (Grace + Hopper)
    * GB10, GB200 (Grace + Blackwell)
    """
    return re.match(r"^g[bh]\d+", gpu_name.lower()) is not None


KNOWN_NVIDIA_GPUS: list[NvidiaGPUInfo] = [
    NvidiaGPUInfo(name="A10", memory=24, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="A16", memory=16, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="A40", memory=48, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="A100", memory=40, compute_capability=(8, 0)),
    NvidiaGPUInfo(name="A100", memory=80, compute_capability=(8, 0)),
    NvidiaGPUInfo(name="A10G", memory=24, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="A4000", memory=16, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="A4500", memory=20, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="A5000", memory=24, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="A6000", memory=48, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="H100", memory=80, compute_capability=(9, 0)),
    NvidiaGPUInfo(name="H100NVL", memory=94, compute_capability=(9, 0)),
    NvidiaGPUInfo(name="H200", memory=141, compute_capability=(9, 0)),
    NvidiaGPUInfo(name="B200", memory=180, compute_capability=(10, 0)),
    NvidiaGPUInfo(name="L4", memory=24, compute_capability=(8, 9)),
    NvidiaGPUInfo(name="L40", memory=48, compute_capability=(8, 9)),
    NvidiaGPUInfo(name="L40S", memory=48, compute_capability=(8, 9)),
    NvidiaGPUInfo(name="P100", memory=16, compute_capability=(6, 0)),
    NvidiaGPUInfo(name="RTX3060", memory=8, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="RTX3060", memory=12, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="RTX3060Ti", memory=8, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="RTX3070Ti", memory=8, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="RTX3080", memory=10, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="RTX3080Ti", memory=12, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="RTX3090", memory=24, compute_capability=(8, 6)),
    NvidiaGPUInfo(name="RTX4090", memory=24, compute_capability=(8, 9)),
    NvidiaGPUInfo(name="RTX6000", memory=24, compute_capability=(7, 5)),
    NvidiaGPUInfo(name="RTX2000Ada", memory=16, compute_capability=(8, 9)),
    NvidiaGPUInfo(name="RTX4000Ada", memory=20, compute_capability=(8, 9)),
    NvidiaGPUInfo(name="RTX6000Ada", memory=48, compute_capability=(8, 9)),
    NvidiaGPUInfo(name="T4", memory=16, compute_capability=(7, 5)),
    NvidiaGPUInfo(name="V100", memory=16, compute_capability=(7, 0)),
    NvidiaGPUInfo(name="V100", memory=32, compute_capability=(7, 0)),
    NvidiaGPUInfo(name="GH200", memory=96, compute_capability=(9, 0)),
]

# For device ids check https://gitlab.freedesktop.org/mesa/libdrm/-/blob/main/data/amdgpu.ids
KNOWN_AMD_GPUS: list[AMDGPUInfo] = [
    AMDGPUInfo(
        name="MI100",
        memory=32,
        architecture=AMDArchitecture.CDNA,
        device_ids=(0x738C,),
    ),
    AMDGPUInfo(
        name="MI210",
        memory=64,
        architecture=AMDArchitecture.CDNA2,
        device_ids=(0x740F,),
    ),
    # TODO: recheck MI250/MI250X device ids on real devices, as the source is ambiguous:
    # 7408, 00, AMD Instinct MI250X
    # 740C, 01, AMD Instinct MI250X / MI250
    AMDGPUInfo(
        name="MI250",
        memory=128,
        architecture=AMDArchitecture.CDNA2,
        device_ids=(0x740C,),
    ),
    AMDGPUInfo(
        name="MI250X",
        memory=128,
        architecture=AMDArchitecture.CDNA2,
        device_ids=(0x7408,),
    ),
    AMDGPUInfo(
        name="MI300A",
        memory=128,
        architecture=AMDArchitecture.CDNA3,
        device_ids=(0x74A0,),
    ),
    AMDGPUInfo(
        name="MI300X",
        memory=192,
        architecture=AMDArchitecture.CDNA3,
        device_ids=(0x74A1, 0x74A9, 0x74B5, 0x74BD),
    ),
    AMDGPUInfo(
        name="MI308X",
        memory=128,
        architecture=AMDArchitecture.CDNA3,
        device_ids=(0x74A2, 0x74B6),
    ),
    AMDGPUInfo(
        name="MI325X",
        memory=288,
        architecture=AMDArchitecture.CDNA3,
        device_ids=(0x74A5,),
    ),
]

KNOWN_TPUS: list[TPUInfo] = [TPUInfo(name=version, memory=0) for version in _TPU_VERSIONS]

KNOWN_INTEL_ACCELERATORS: list[IntelAcceleratorInfo] = [
    IntelAcceleratorInfo(name="Gaudi", memory=32),  # HL-205
    IntelAcceleratorInfo(name="Gaudi2", memory=96),  # HL-225
    IntelAcceleratorInfo(name="Gaudi3", memory=128),
]

KNOWN_TENSTORRENT_ACCELERATORS: list[TenstorrentAcceleratorInfo] = [
    TenstorrentAcceleratorInfo(name="n150", memory=12),
    TenstorrentAcceleratorInfo(name="n300", memory=24),
]

KNOWN_ACCELERATORS: list[
    Union[NvidiaGPUInfo, AMDGPUInfo, TPUInfo, IntelAcceleratorInfo, TenstorrentAcceleratorInfo]
] = (
    KNOWN_NVIDIA_GPUS
    + KNOWN_AMD_GPUS
    + KNOWN_TPUS
    + KNOWN_INTEL_ACCELERATORS
    + KNOWN_TENSTORRENT_ACCELERATORS
)
