from typing import Union

import pytest

from gpuhunt._internal.models import AcceleratorVendor, CatalogItem, Optional, RawCatalogItem


@pytest.mark.parametrize(
    ["gpu_vendor", "gpu_count", "gpu_name", "expected_gpu_vendor"],
    [
        pytest.param(None, None, None, None, id="none-gpu"),
        pytest.param(None, 0, None, None, id="zero-gpu"),
        pytest.param(None, 1, "A100", "nvidia", id="one-gpu-not-tpu"),
        pytest.param(None, 1, "tpu-v3", "google", id="one-gpu-tpu"),
        pytest.param("amd", 0, "tpu-v3", "amd", id="vendor-already-set"),  # no heuristic used
        pytest.param(AcceleratorVendor.AMD, 1, "MI300X", "amd", id="cast-enum-to-string"),
    ],
)
def test_raw_catalog_item_gpu_vendor_heuristic(
    gpu_vendor: Union[AcceleratorVendor, str, None],
    gpu_count: Optional[int],
    gpu_name: Optional[str],
    expected_gpu_vendor: Optional[str],
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


@pytest.mark.parametrize(
    ["gpu_vendor", "gpu_count", "gpu_name", "expected_gpu_vendor"],
    [
        pytest.param(None, None, None, None, id="none-gpu"),
        pytest.param(None, 0, None, None, id="zero-gpu"),
        pytest.param(None, 1, "A100", AcceleratorVendor.NVIDIA, id="one-gpu-not-tpu"),
        pytest.param(None, 1, "tpu-v3", AcceleratorVendor.GOOGLE, id="one-gpu-tpu"),
        # no heuristic used
        pytest.param(
            AcceleratorVendor.AMD, 0, "tpu-v3", AcceleratorVendor.AMD, id="vendor-already-set"
        ),
        pytest.param("amd", 1, "MI300X", AcceleratorVendor.AMD, id="cast-string-to-enum"),
    ],
)
def test_catalog_item_gpu_vendor_heuristic(
    gpu_vendor: Union[AcceleratorVendor, str, None],
    gpu_count: Optional[int],
    gpu_name: Optional[str],
    expected_gpu_vendor: Optional[str],
):
    dct = {}
    if gpu_vendor is not None:
        dct["gpu_vendor"] = gpu_vendor
    if gpu_count is not None:
        dct["gpu_count"] = gpu_count
    if gpu_name is not None:
        dct["gpu_name"] = gpu_name

    item = CatalogItem.from_dict(dct)

    assert item.gpu_vendor == expected_gpu_vendor
