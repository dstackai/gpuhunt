import warnings

from gpuhunt._internal import constraints as constraints
from gpuhunt._internal.catalog import Catalog as Catalog
from gpuhunt._internal.constraints import (
    KNOWN_ACCELERATORS as KNOWN_ACCELERATORS,
    KNOWN_AMD_GPUS as KNOWN_AMD_GPUS,
    KNOWN_INTEL_ACCELERATORS as KNOWN_INTEL_ACCELERATORS,
    KNOWN_NVIDIA_GPUS as KNOWN_NVIDIA_GPUS,
    KNOWN_TENSTORRENT_ACCELERATORS as KNOWN_TENSTORRENT_ACCELERATORS,
    KNOWN_TPUS as KNOWN_TPUS,
    correct_gpu_memory_gib as correct_gpu_memory_gib,
    is_nvidia_superchip as is_nvidia_superchip,
    matches as matches,
)
from gpuhunt._internal.default import (
    default_catalog as default_catalog,
    query as query,
)
from gpuhunt._internal.models import (
    AcceleratorInfo as AcceleratorInfo,
    AcceleratorVendor as AcceleratorVendor,
    AMDGPUInfo as AMDGPUInfo,
    CatalogItem as CatalogItem,
    CPUArchitecture as CPUArchitecture,
    IntelAcceleratorInfo as IntelAcceleratorInfo,
    NvidiaGPUInfo as NvidiaGPUInfo,
    QueryFilter as QueryFilter,
    RawCatalogItem as RawCatalogItem,
    TenstorrentAcceleratorInfo as TenstorrentAcceleratorInfo,
    TPUInfo as TPUInfo,
)

# Deprecated aliases
GPUInfo: type[NvidiaGPUInfo]
KNOWN_GPUS: list[NvidiaGPUInfo]


def _warn_renamed(old: str, new: str) -> None:
    warnings.warn(
        f"{old} has been renamed to {new}, the old name is deprecated and will be removed.",
        DeprecationWarning,
        stacklevel=2,
    )


def __getattr__(name):
    if name == "GPUInfo":
        _warn_renamed("GPUInfo", "NvidiaGPUInfo")
        return NvidiaGPUInfo
    if name == "KNOWN_GPUS":
        _warn_renamed("KNOWN_GPUS", "KNOWN_NVIDIA_GPUS")
        return KNOWN_NVIDIA_GPUS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
