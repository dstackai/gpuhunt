from typing import List

import pytest

from gpuhunt import CatalogItem, QueryFilter
from gpuhunt._internal.constraints import matches


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
        disk_size=None,
    )


@pytest.fixture
def cpu_items() -> List[CatalogItem]:
    datacrunch = CatalogItem(
        instance_name="CPU.120V.480G",
        location="ICE-01",
        price=3.0,
        cpu=120,
        memory=480.0,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=0.0,
        spot=False,
        provider="datacrunch",
        disk_size=None,
    )
    nebius = CatalogItem(
        instance_name="standard-v2",
        location="eu-north1-c",
        price=1.4016,
        cpu=48,
        memory=288.0,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        spot=False,
        provider="nebius",
        disk_size=None,
    )
    return [datacrunch, nebius]


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
            disk_size=None,
        )
        assert matches(item, QueryFilter(gpu_name=["RTX3060TI"]))

    def test_provider(self, cpu_items):
        assert matches(cpu_items[0], QueryFilter(provider=["datacrunch"]))
        assert matches(cpu_items[1], QueryFilter(provider=["nebius"]))
