import pytest

from gpuhunt._internal.constraints import KNOWN_AMD_GPUS
from gpuhunt._internal.models import AcceleratorVendor, AMDArchitecture, CatalogItem, QueryFilter


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


@pytest.mark.parametrize(
    ["query_filter", "expected"],
    [
        pytest.param(QueryFilter(), "QueryFilter()", id="empty"),
        pytest.param(QueryFilter(min_cpu=4), "QueryFilter(min_cpu=4)", id="single-field"),
        pytest.param(
            QueryFilter(max_price=1.2, min_cpu=4),
            "QueryFilter(min_cpu=4, max_price=1.2)",
            id="fields-in-declaration-order",
        ),
    ],
)
def test_query_filter_repr(query_filter: QueryFilter, expected: str):
    assert repr(query_filter) == expected


class TestCatalogItem:
    def make_item(self, **kwargs) -> CatalogItem:
        defaults = dict(
            provider="test",
            instance_name="test-instance",
            location="eu-west-1",
            price=1.0,
            cpu=1,
            memory=32.0,
            gpu_count=1,
            gpu_name="A100",
            gpu_memory=40.0,
            gpu_vendor=AcceleratorVendor.NVIDIA,
            spot=False,
            disk_size=None,
        )
        return CatalogItem(**{**defaults, **kwargs})

    def test_gpu_item(self):
        item = self.make_item()
        assert item.gpu_name == "A100"

    def test_cpu_item_without_gpu_details(self):
        item = self.make_item(gpu_count=0, gpu_name=None, gpu_memory=None, gpu_vendor=None)
        assert item.gpu_count == 0

    @pytest.mark.parametrize("field", ["gpu_vendor", "gpu_name", "gpu_memory"])
    def test_gpu_item_requires_gpu_details(self, field: str):
        with pytest.raises(ValueError, match=f"{field} is required"):
            self.make_item(**{field: None})
