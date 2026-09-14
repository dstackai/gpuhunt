import pytest

from gpuhunt._internal.models import QueryFilter
from gpuhunt.providers.vastai import VastAIProvider, bundles_url, compute_cap


def make_offer(**overrides) -> dict:
    offer = {
        "id": 1,
        "cpu_cores": 32,
        "cpu_cores_effective": 4.0,
        "cpu_ram": 7859,
        "disk_space": 100.0,
        "storage_cost": 0.1,
        "dph_base": 0.2,
        "geolocation": "Bulgaria, BG",
        "gpu_name": "RTX 3090",
        "gpu_ram": 24576,
        "num_gpus": 1,
        "rentable": True,
        "rented": False,
    }
    offer.update(overrides)
    return offer


class TestGet:
    def test_keeps_fractional_memory(self, requests_mock):
        requests_mock.post(bundles_url, json={"offers": [make_offer()]})

        offers = VastAIProvider().get()

        assert [offer.memory for offer in offers] == [0.98]

    def test_skips_offers_with_less_than_one_cpu(self, requests_mock):
        requests_mock.post(bundles_url, json={"offers": [make_offer(cpu_cores_effective=0.5)]})

        assert VastAIProvider().get() == []

    def test_skips_offers_without_memory(self, requests_mock):
        requests_mock.post(bundles_url, json={"offers": [make_offer(cpu_ram=0)]})

        assert VastAIProvider().get() == []


def test_make_filters_defaults_to_datacenter_only():
    filters = VastAIProvider(community_cloud=False).make_filters(QueryFilter())
    assert filters["datacenter"] == {"eq": True}
    assert "external" not in filters


def test_make_filters_does_not_constrain_scope_when_community_cloud_enabled():
    filters = VastAIProvider(community_cloud=True).make_filters(QueryFilter())
    assert "datacenter" not in filters
    assert "external" not in filters


@pytest.mark.parametrize(
    ["cc", "expected"],
    [
        pytest.param((7, 0), "700", id="7.0"),
        pytest.param((7, 5), "750", id="7.5"),
    ],
)
def test_compute_cap(cc: tuple[int, int], expected: str):
    assert compute_cap(cc) == expected
