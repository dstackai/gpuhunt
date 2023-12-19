from typing import List

import pytest

from gpuhunt import QueryFilter
from gpuhunt._internal.models import RawCatalogItem
from gpuhunt.providers.tensordock import TensorDockProvider


@pytest.fixture
def specs() -> dict:
    return {
        "cpu": {"amount": 256, "price": 0.003, "type": "Intel Xeon Platinum 8352Y"},
        "gpu": {
            "l40-pcie-48gb": {
                "amount": 8,
                "gtx": False,
                "pcie": True,
                "price": 1.05,
                "rtx": False,
                "vram": 48,
            }
        },
        "ram": {"amount": 1495, "price": 0.002},
        "storage": {"amount": 10252, "price": 5e-05},
    }


class TestTensorDockMinimalConfiguration:
    def test_no_requirements(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(QueryFilter(), specs, "", "")
        assert offers == make_offers(specs, cpu=16, memory=96, disk_size=96, gpu_count=1)

    def test_min_cpu_no_balance(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(
            QueryFilter(min_cpu=4), specs, "", "", balance_resources=False
        )
        assert offers == make_offers(specs, cpu=4, memory=96, disk_size=96, gpu_count=1)

    def test_min_cpu(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(QueryFilter(min_cpu=4), specs, "", "")
        assert offers == make_offers(specs, cpu=16, memory=96, disk_size=96, gpu_count=1)

    def test_too_many_min_cpu(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(QueryFilter(min_cpu=1000), specs, "", "")
        assert offers == []

    def test_min_memory_no_balance(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(
            QueryFilter(min_memory=3), specs, "", "", balance_resources=False
        )
        assert offers == make_offers(specs, cpu=2, memory=4, disk_size=48, gpu_count=1)

    def test_min_memory(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(QueryFilter(min_memory=3), specs, "", "")
        assert offers == make_offers(specs, cpu=16, memory=96, disk_size=96, gpu_count=1)

    def test_too_large_min_memory(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(QueryFilter(min_memory=2000), specs, "", "")
        assert offers == []

    def test_min_gpu_count(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(QueryFilter(min_gpu_count=2), specs, "", "")
        assert offers == make_offers(specs, cpu=32, memory=192, disk_size=192, gpu_count=2)

    def test_min_no_gpu(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(QueryFilter(max_gpu_count=0), specs, "", "")
        assert offers == []

    def test_min_total_gpu_memory(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(
            QueryFilter(min_total_gpu_memory=100), specs, "", ""
        )
        assert offers == make_offers(specs, cpu=48, memory=288, disk_size=288, gpu_count=3)

    def test_controversial_gpu(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(
            QueryFilter(min_total_gpu_memory=100, max_gpu_count=2), specs, "", ""
        )
        assert offers == []

    def test_all_cpu_all_gpu(self, specs: dict):
        offers = TensorDockProvider.optimize_offers(
            QueryFilter(min_cpu=256, min_gpu_count=1), specs, "", ""
        )
        assert offers == make_offers(specs, cpu=256, memory=768, disk_size=768, gpu_count=8)


def make_offers(
    specs: dict, cpu: int, memory: float, disk_size: float, gpu_count: int
) -> List[RawCatalogItem]:
    gpu = list(specs["gpu"].values())[0]
    price = cpu * specs["cpu"]["price"]
    price += memory * specs["ram"]["price"]
    price += disk_size * specs["storage"]["price"]
    price += gpu_count * gpu["price"]
    return [
        RawCatalogItem(
            instance_name="",
            location="",
            price=round(price, 5),
            cpu=cpu,
            memory=memory,
            gpu_count=gpu_count,
            gpu_name="L40",
            gpu_memory=gpu["vram"],
            spot=False,
            disk_size=disk_size,
        )
    ]
