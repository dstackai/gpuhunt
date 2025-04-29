# ruff: noqa: F401
import warnings

from gpuhunt._internal.catalog import Catalog
from gpuhunt._internal.constraints import (
    KNOWN_ACCELERATORS,
    KNOWN_AMD_GPUS,
    KNOWN_INTEL_ACCELERATORS,
    KNOWN_NVIDIA_GPUS,
    KNOWN_TENSTORRENT_ACCELERATORS,
    KNOWN_TPUS,
    correct_gpu_memory_gib,
    is_nvidia_superchip,
    matches,
)
from gpuhunt._internal.default import default_catalog, query
from gpuhunt._internal.models import (
    AcceleratorInfo,
    AcceleratorVendor,
    AMDGPUInfo,
    CatalogItem,
    CPUArchitecture,
    IntelAcceleratorInfo,
    NvidiaGPUInfo,
    QueryFilter,
    RawCatalogItem,
    TenstorrentAcceleratorInfo,
    TPUInfo,
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
