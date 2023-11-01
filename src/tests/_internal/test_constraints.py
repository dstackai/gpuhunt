import pytest

from gpuhunt import CatalogItem, QueryFilter
from gpuhunt._internal.constraints import fill_missing, matches


@pytest.fixture
def item() -> CatalogItem:
    return CatalogItem(
        instance_name="large",
        location="us-east-1",
        price=1.2,
        cpu=16,
        memory=64.0,
        gpu_count=1,
        gpu_name="A100",
        gpu_memory=40.0,
        spot=False,
        provider="aws",
    )


class TestMatches:
    def test_empty(self, item: CatalogItem):
        assert matches(item, QueryFilter())

    def test_cpu(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_cpu=16))
        assert matches(item, QueryFilter(max_cpu=16))
        assert not matches(item, QueryFilter(min_cpu=32))
        assert not matches(item, QueryFilter(max_cpu=8))

    def test_memory(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_memory=64.0))
        assert matches(item, QueryFilter(max_memory=64.0))
        assert not matches(item, QueryFilter(min_memory=128.0))
        assert not matches(item, QueryFilter(max_memory=32.0))

    def test_gpu_count(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_gpu_count=1))
        assert matches(item, QueryFilter(max_gpu_count=1))
        assert not matches(item, QueryFilter(min_gpu_count=2))
        assert not matches(item, QueryFilter(max_gpu_count=0))

    def test_gpu_memory(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_gpu_memory=40.0))
        assert matches(item, QueryFilter(max_gpu_memory=40.0))
        assert not matches(item, QueryFilter(min_gpu_memory=80.0))
        assert not matches(item, QueryFilter(max_gpu_memory=20.0))

    def test_gpu_name(self, item: CatalogItem):
        assert matches(item, QueryFilter(gpu_name=["A100"]))
        assert not matches(item, QueryFilter(gpu_name=["A10"]))

    def test_total_gpu_memory(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_total_gpu_memory=40.0))
        assert matches(item, QueryFilter(max_total_gpu_memory=40.0))
        assert not matches(item, QueryFilter(min_total_gpu_memory=80.0))
        assert not matches(item, QueryFilter(max_total_gpu_memory=20.0))

    def test_price(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_price=1.2))
        assert matches(item, QueryFilter(max_price=1.2))
        assert not matches(item, QueryFilter(min_price=1.3))
        assert not matches(item, QueryFilter(max_price=1.1))

    def test_spot(self, item: CatalogItem):
        assert matches(item, QueryFilter(spot=False))
        assert not matches(item, QueryFilter(spot=True))

    def test_compute_capability(self, item: CatalogItem):
        assert matches(item, QueryFilter(min_compute_capability=(8, 0)))
        assert matches(item, QueryFilter(max_compute_capability=(8, 0)))
        assert not matches(item, QueryFilter(min_compute_capability=(8, 1)))
        assert not matches(item, QueryFilter(max_compute_capability=(7, 9)))

    def test_ti_gpu(self):
        item = CatalogItem(
            instance_name="large",
            location="us-east-1",
            price=1.2,
            cpu=16,
            memory=64.0,
            gpu_count=1,
            gpu_name="RTX3060Ti",  # case-sensitive
            gpu_memory=8.0,
            spot=False,
            provider="aws",
        )
        assert matches(item, QueryFilter(gpu_name=["RTX3060TI"]))


class TestFillMissing:
    def test_empty(self):
        assert fill_missing(QueryFilter(), memory_per_core=4) == QueryFilter()

    def test_from_cpu(self):
        assert fill_missing(QueryFilter(min_cpu=2), memory_per_core=4) == QueryFilter(
            min_cpu=2,
            min_memory=8,
        )

    def test_from_memory(self):
        assert fill_missing(QueryFilter(min_memory=6), memory_per_core=4) == QueryFilter(
            min_memory=6,
            min_cpu=2,
        )

    def test_from_total_gpu_memory(self):
        assert fill_missing(
            QueryFilter(min_total_gpu_memory=24), memory_per_core=4
        ) == QueryFilter(
            min_total_gpu_memory=24,
            min_memory=48,
            min_disk_size=54,
            min_cpu=12,
        )

    def test_from_gpu_memory(self):
        assert fill_missing(QueryFilter(min_gpu_memory=16), memory_per_core=4) == QueryFilter(
            min_gpu_memory=16,
            min_memory=32,
            min_disk_size=46,
            min_cpu=8,
        )

    def test_from_gpu_count(self):
        assert fill_missing(QueryFilter(min_gpu_count=2), memory_per_core=4) == QueryFilter(
            min_gpu_count=2,
            min_memory=32,  # minimal GPU has 8 GB of memory
            min_disk_size=46,
            min_cpu=8,
        )

    def test_from_gpu_name(self):
        assert fill_missing(QueryFilter(gpu_name=["A100"]), memory_per_core=4) == QueryFilter(
            gpu_name=["A100"],
            min_memory=80,
            min_disk_size=70,
            min_cpu=20,
        )

    def test_from_compute_capability(self):
        assert fill_missing(
            QueryFilter(min_compute_capability=(9, 0)), memory_per_core=4
        ) == QueryFilter(
            min_compute_capability=(9, 0),
            min_memory=160,
            min_disk_size=110,
            min_cpu=40,
        )

    def test_from_gpu_name_and_gpu_memory(self):
        assert fill_missing(
            QueryFilter(gpu_name=["A100"], min_gpu_memory=80), memory_per_core=4
        ) == QueryFilter(
            gpu_name=["A100"],
            min_gpu_memory=80,
            min_memory=160,
            min_disk_size=110,
            min_cpu=40,
        )
