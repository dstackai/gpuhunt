import pytest

from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers.aws import _get_gpu_memory_gib, _make_offer, _PricingRow


def pricing_row(**kwargs) -> _PricingRow:
    defaults = dict(
        instance_name="g5.xlarge",
        location="us-east-1",
        price=1.006,
        cpu=4,
        memory=16.0,
        gpu_count=1,
    )
    return _PricingRow(**{**defaults, **kwargs})


class TestMakeOffer:
    def test_gpu_instance(self):
        offer = _make_offer(pricing_row(), gpus={"g5.xlarge": ("A10G", 24.0)})
        assert offer.gpu_count == 1
        assert offer.gpu_vendor == AcceleratorVendor.NVIDIA
        assert offer.gpu_name == "A10G"
        assert offer.gpu_memory == 24.0

    def test_gpu_instance_without_gpu_details_becomes_cpu_instance(self):
        offer = _make_offer(pricing_row(), gpus={})
        assert offer.gpu_count == 0
        assert offer.gpu_vendor is None
        assert offer.gpu_name is None
        assert offer.gpu_memory is None

    def test_cpu_instance(self):
        offer = _make_offer(
            pricing_row(instance_name="c5.large", gpu_count=0),
            gpus={"g5.xlarge": ("A10G", 24.0)},
        )
        assert offer.gpu_count == 0
        assert offer.gpu_vendor is None
        assert offer.gpu_name is None


class TestGetGpuMemoryGib:
    @pytest.mark.parametrize(
        ("name", "memory_mib", "expected"),
        [
            ("B300", 275040, 270),
            ("L4", 22888, 24),
            ("RTX PRO Server 6000", 98304, 96),
        ],
    )
    def test_full_gpu_configuration(self, name, memory_mib, expected):
        assert _get_gpu_memory_gib(name, memory_mib) == expected
