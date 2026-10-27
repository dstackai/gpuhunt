from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers.aws import _make_offer, _PricingRow


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
