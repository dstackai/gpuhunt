from typing import Union

import pytest

from gpuhunt._internal.constraints import KNOWN_AMD_GPUS
from gpuhunt._internal.models import (
    AcceleratorVendor,
    AMDArchitecture,
    CatalogItem,
    Optional,
    RawCatalogItem,
)

NVIDIA = AcceleratorVendor.NVIDIA
GOOGLE = AcceleratorVendor.GOOGLE
AMD = AcceleratorVendor.AMD


@pytest.mark.parametrize(
    ["gpu_count", "gpu_vendor", "gpu_name", "expected_gpu_vendor", "expected_gpu_name"],
    [
        pytest.param(None, None, None, None, None, id="none-gpu"),
        pytest.param(0, None, None, None, None, id="zero-gpu"),
        pytest.param(1, None, "A100", "nvidia", "A100", id="one-gpu"),
        pytest.param(1, None, "tpu-v3", "google", "v3", id="one-tpu-vendor-not-set"),
        pytest.param(1, "google", "tpu-v5p", "google", "v5p", id="one-tpu-vendor-is-set"),
        pytest.param(1, AMD, "MI300X", "amd", "MI300X", id="cast-enum-to-string"),
    ],
)
def test_raw_catalog_item_gpu_vendor_heuristic(
    gpu_count: Optional[int],
    gpu_vendor: Union[AcceleratorVendor, str, None],
    gpu_name: Optional[str],
    expected_gpu_vendor: Optional[str],
    expected_gpu_name: Optional[str],
):
    dct = {}
    if gpu_vendor is not None:
        dct["gpu_vendor"] = gpu_vendor
    if gpu_count is not None:
        dct["gpu_count"] = gpu_count
    if gpu_name is not None:
        dct["gpu_name"] = gpu_name

    item = RawCatalogItem.from_dict(dct)

    assert item.gpu_vendor == expected_gpu_vendor
    assert item.gpu_name == expected_gpu_name


@pytest.mark.parametrize(
    ["gpu_count", "gpu_vendor", "gpu_name", "expected_gpu_vendor"],
    [
        pytest.param(None, None, None, None, id="none-gpu"),
        pytest.param(0, None, None, None, id="zero-gpu"),
        pytest.param(1, None, None, NVIDIA, id="one-gpu-no-name"),
        pytest.param(1, None, "v3", NVIDIA, id="one-gpu-with-any-name"),
        pytest.param(1, "amd", "MI300X", AMD, id="cast-string-to-enum"),
    ],
)
def test_catalog_item_gpu_vendor_heuristic(
    gpu_count: Optional[int],
    gpu_vendor: Union[AcceleratorVendor, str, None],
    gpu_name: Optional[str],
    expected_gpu_vendor: Optional[str],
):
    item = CatalogItem(
        instance_name="test-instance",
        location="eu-west-1",
        price=1.0,
        cpu=1,
        memory=32.0,
        gpu_vendor=gpu_vendor,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory=8.0,
        spot=False,
        disk_size=100.0,
        provider="test",
    )

    assert item.gpu_vendor == expected_gpu_vendor


@pytest.mark.parametrize(
    ["model", "architecture", "expected_memory"],
    [
        pytest.param("MI325X", AMDArchitecture.CDNA3, 288, id="MI325X"),
        pytest.param("MI308X", AMDArchitecture.CDNA3, 128, id="MI308X"),
        pytest.param("MI300X", AMDArchitecture.CDNA3, 192, id="MI300X"),
        pytest.param("MI300A", AMDArchitecture.CDNA3, 128, id="MI300A"),
        pytest.param("MI250X", AMDArchitecture.CDNA2, 128, id="MI250X"),
        pytest.param("MI250", AMDArchitecture.CDNA2, 128, id="MI250"),
        pytest.param("MI210", AMDArchitecture.CDNA2, 64, id="MI210"),
        pytest.param("MI100", AMDArchitecture.CDNA, 32, id="MI100"),
    ],
)
def test_amd_gpu_architecture(model: str, architecture: AMDArchitecture, expected_memory: int):
    for gpu in KNOWN_AMD_GPUS:
        if gpu.name == model:
            assert gpu.architecture == architecture
            assert gpu.memory == expected_memory
            return
    # If we get here, the test should fail since we could not find the GPU in our known list.
    assert False
