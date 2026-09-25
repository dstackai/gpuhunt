import copy
import logging

import pytest
import requests

import gpuhunt.providers.daytona as daytona_module
from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers.daytona import API_URL, GPU_COUNTS, GPU_MAP, DaytonaProvider

PAYLOAD = {
    "generatedAt": "2026-09-23T08:43:09Z",
    "currency": "USD",
    "gpus": [
        {
            "type": "B300",
            "onDemandPricePerHour": 7.1,
            "spotPricePerHour": 4.08,
            "onDemandPricePerSecond": 0.0019722222222222222,
            "spotPricePerSecond": 0.0011333333333333334,
        },
        {
            "type": "B200",
            "onDemandPricePerHour": 6.25,
            "spotPricePerHour": 3.59,
            "onDemandPricePerSecond": 0.001736111111111111,
            "spotPricePerSecond": 0.0009972222222222223,
        },
        {
            "type": "MI355X",
            "onDemandPricePerHour": 5.99,
            "spotPricePerHour": 3.44,
            "onDemandPricePerSecond": 0.0016638888888888888,
            "spotPricePerSecond": 0.0009555555555555555,
        },
        {
            "type": "H200",
            "onDemandPricePerHour": 4.54,
            "spotPricePerHour": 2.61,
            "onDemandPricePerSecond": 0.001261,
            "spotPricePerSecond": 0.000725,
        },
        {
            "type": "H100",
            "onDemandPricePerHour": 3.95,
            "spotPricePerHour": 2.27,
            "onDemandPricePerSecond": 0.001097,
            "spotPricePerSecond": 0.0006305555555555555,
        },
        {
            "type": "RTX-PRO-6000",
            "onDemandPricePerHour": 3.03,
            "spotPricePerHour": 1.74,
            "onDemandPricePerSecond": 0.0008416666666666667,
            "spotPricePerSecond": 0.00048333333333333334,
        },
        {
            "type": "RTX-5090",
            "onDemandPricePerHour": 1.29,
            "spotPricePerHour": 0.74,
            "onDemandPricePerSecond": 0.00035833333333333333,
            "spotPricePerSecond": 0.00020555555555555556,
        },
        {
            "type": "RTX-4090",
            "onDemandPricePerHour": 0.99,
            "spotPricePerHour": 0.57,
            "onDemandPricePerSecond": 0.000275,
            "spotPricePerSecond": 0.00015833333333333332,
        },
    ],
    "resources": {
        "onDemand": {
            "vcpuPerHour": 0.0504,
            "memoryGiBPerHour": 0.0162,
            "diskGiBPerHour": 0.000108,
        },
        "spot": {
            "vcpuPerHour": 0.03,
            "memoryGiBPerHour": 0.0093,
            "diskGiBPerHour": 0.000062,
        },
    },
    "notes": ["GPU rates are per GPU. Sandbox vCPU, memory, and disk are billed separately."],
}


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self.payload


@pytest.fixture
def requested_urls(monkeypatch) -> list[tuple[str, float]]:
    requested = []

    def fake_get(url, timeout):
        requested.append((url, timeout))
        return FakeResponse(copy.deepcopy(PAYLOAD))

    monkeypatch.setattr(daytona_module.requests, "get", fake_get)
    return requested


def test_offers_cover_every_gpu_count_and_capacity_type(requested_urls):
    offers = DaytonaProvider().get()

    assert len(offers) == len(PAYLOAD["gpus"]) * len(GPU_COUNTS) * 2
    assert [(url, timeout) for url, timeout in requested_urls] == [(API_URL, 10.0)]
    assert offers == sorted(offers, key=lambda o: o.price)
    assert {o.gpu_name for o in offers} == {name for name, _, _, _ in GPU_MAP.values()}
    assert {o.gpu_count for o in offers} == set(GPU_COUNTS)
    assert {o.spot for o in offers} == {False, True}
    assert all(o.provider == "daytona" and o.location == "us" for o in offers)


def test_prices_combine_gpu_and_resource_rates_per_capacity_type(requested_urls):
    offers = DaytonaProvider().get()

    def find(gpu_name: str, gpu_count: int, spot: bool):
        return next(
            o
            for o in offers
            if o.gpu_name == gpu_name and o.gpu_count == gpu_count and o.spot is spot
        )

    # 3.95 GPU + 8 vCPU * 0.0504 + 100 GB * 0.0162 + 256 GB disk * 0.000108
    h100_on_demand = find("H100", 1, spot=False)
    assert h100_on_demand.price == 6.000848
    assert (h100_on_demand.cpu, h100_on_demand.memory, h100_on_demand.disk_size) == (
        8,
        100.0,
        256.0,
    )
    assert h100_on_demand.gpu_memory == 80.0
    assert h100_on_demand.gpu_vendor == AcceleratorVendor.NVIDIA

    # Spot discounts the resources too: 2.27 + 8 * 0.03 + 100 * 0.0093 + 256 * 0.000062
    h100_spot = find("H100", 1, spot=True)
    assert h100_spot.price == 3.455872

    # Consumer cards carry 50 GB RAM per GPU.
    rtx4090_on_demand = find("RTX4090", 1, spot=False)
    assert rtx4090_on_demand.price == 2.230848
    assert rtx4090_on_demand.memory == 50.0

    # Resources scale linearly with the GPU count.
    h100_8x = find("H100", 8, spot=False)
    assert (h100_8x.cpu, h100_8x.memory, h100_8x.disk_size) == (64, 800.0, 2048.0)
    assert h100_8x.price == 48.006784

    mi355x = find("MI355X", 1, spot=False)
    assert mi355x.gpu_vendor == AcceleratorVendor.AMD
    assert mi355x.gpu_memory == 288.0


def test_unknown_gpu_type_is_skipped_with_warning(monkeypatch, caplog):
    payload = copy.deepcopy(PAYLOAD)
    payload["gpus"].append(
        {"type": "B400", "onDemandPricePerHour": 9.99, "spotPricePerHour": 5.99}
    )
    monkeypatch.setattr(daytona_module.requests, "get", lambda url, timeout: FakeResponse(payload))

    with caplog.at_level(logging.WARNING):
        offers = DaytonaProvider().get()

    assert len(offers) == len(PAYLOAD["gpus"]) * len(GPU_COUNTS) * 2
    assert "B400" in caplog.text


def test_http_error_propagates(monkeypatch):
    monkeypatch.setattr(
        daytona_module.requests, "get", lambda url, timeout: FakeResponse({}, status_code=500)
    )

    with pytest.raises(requests.HTTPError):
        DaytonaProvider().get()


@pytest.mark.parametrize(
    "payload",
    [
        "not a dict",
        {},
        {"gpus": "not a list", "resources": {"onDemand": {}, "spot": {}}},
        {"gpus": [], "resources": {"onDemand": {}}},
    ],
)
def test_malformed_payload_is_rejected(monkeypatch, payload):
    monkeypatch.setattr(daytona_module.requests, "get", lambda url, timeout: FakeResponse(payload))

    with pytest.raises(ValueError):
        DaytonaProvider().get()
