import pytest
from requests import RequestException

from gpuhunt._internal.models import RawCatalogItem
from gpuhunt.providers import runpod as runpod_module
from gpuhunt.providers.runpod import RunpodProvider, _cpu_size_ladder


def test_cpu_size_ladder():
    assert _cpu_size_ladder(2, 32) == [2, 4, 8, 16, 32]
    assert _cpu_size_ladder(3, 20) == [3, 6, 12, 20]


def test_make_cpu_catalog_items():
    provider = object.__new__(RunpodProvider)
    cpu_flavors = [
        {
            "id": "cpu3g",
            "minVcpu": 2,
            "maxVcpu": 32,
            "ramMultiplier": 4,
            "diskLimitPerVcpu": 10,
            "specifics": {
                "stockStatus": "High",
                "securePrice": 0.08,
                "slsPrice": 0.1,
            },
        }
    ]

    items = provider._make_cpu_catalog_items("AP-JP-1", cpu_flavors)

    assert [item.instance_name for item in items] == [
        "cpu3g-2-8",
        "cpu3g-4-16",
        "cpu3g-8-32",
        "cpu3g-16-64",
        "cpu3g-32-128",
    ]
    assert [item.cpu for item in items] == [2, 4, 8, 16, 32]
    assert [item.memory for item in items] == [8, 16, 32, 64, 128]
    assert [item.location for item in items] == ["AP-JP-1"] * 5
    assert [item.spot for item in items] == [False] * 5
    assert [item.gpu_count for item in items] == [0] * 5
    assert [item.flags for item in items] == [["runpod-cpu"]] * 5
    assert [item.disk_size for item in items] == [20.0, 40.0, 80.0, 160.0, 320.0]
    assert items[0].price == pytest.approx(0.08)
    assert items[-1].price == pytest.approx(1.28)
    assert items[0].provider_data == {}
    assert items[-1].provider_data == {}


def test_make_cpu_catalog_items_skips_invalid_flavors():
    provider = object.__new__(RunpodProvider)
    cpu_flavors = [
        {
            "id": "cpu3c",
            "minVcpu": 2,
            "maxVcpu": 32,
            "ramMultiplier": 2,
            "diskLimitPerVcpu": 10,
            "specifics": {
                "stockStatus": None,
                "securePrice": 0.06,
                "slsPrice": 0.072,
            },
        },
        {
            "id": "cpu3m",
            "minVcpu": 2,
            "maxVcpu": 32,
            "ramMultiplier": 8,
            "diskLimitPerVcpu": 10,
            "specifics": {
                "stockStatus": "High",
                "securePrice": None,
                "slsPrice": 0,
            },
        },
        {
            "id": "cpu5c",
            "minVcpu": None,
            "maxVcpu": 32,
            "ramMultiplier": 2,
            "diskLimitPerVcpu": 15,
            "specifics": {
                "stockStatus": "High",
                "securePrice": 0.07,
                "slsPrice": 0.084,
            },
        },
        {
            "id": "cpu5m",
            "minVcpu": 16,
            "maxVcpu": 8,
            "ramMultiplier": 8,
            "diskLimitPerVcpu": 10,
            "specifics": {
                "stockStatus": "High",
                "securePrice": 0.13,
                "slsPrice": 0,
            },
        },
    ]

    assert provider._make_cpu_catalog_items("AP-JP-1", cpu_flavors) == []


def test_fetch_cpu_offers_handles_partial_datacenter_failures(monkeypatch):
    provider = object.__new__(RunpodProvider)

    def fake_make_request(payload):
        assert payload["query"] == runpod_module.cpu_data_centers_query
        return {
            "data": {
                "dataCenters": [
                    {"id": "DC-ERR", "listed": True},
                    {"id": "DC-OK", "listed": True},
                    {"id": "DC-SKIP", "listed": False},
                ]
            }
        }

    def fake_get_cpu_flavors(dc_id: str):
        if dc_id == "DC-ERR":
            raise RequestException("boom")
        return [
            {
                "id": "cpu3c",
                "minVcpu": 2,
                "maxVcpu": 32,
                "ramMultiplier": 2,
                "diskLimitPerVcpu": 10,
                "specifics": {
                    "stockStatus": "High",
                    "securePrice": 0.06,
                    "slsPrice": 0.072,
                },
            }
        ]

    monkeypatch.setattr(runpod_module, "_make_request", fake_make_request)
    monkeypatch.setattr(provider, "_get_cpu_flavors", fake_get_cpu_flavors)

    items = provider._fetch_cpu_offers()

    assert len(items) == 5
    assert {item.location for item in items} == {"DC-OK"}
    assert {tuple(item.flags) for item in items} == {("runpod-cpu",)}
    assert {item.disk_size for item in items} == {20.0, 40.0, 80.0, 160.0, 320.0}
    assert {item.instance_name for item in items} == {
        "cpu3c-2-4",
        "cpu3c-4-8",
        "cpu3c-8-16",
        "cpu3c-16-32",
        "cpu3c-32-64",
    }


def test_fetch_offers_appends_cpu_items(monkeypatch):
    provider = object.__new__(RunpodProvider)
    cpu_item = RawCatalogItem(
        instance_name="cpu3g-2-8",
        location="AP-JP-1",
        price=0.08,
        cpu=2,
        memory=8,
        gpu_count=0,
        gpu_name=None,
        gpu_memory=None,
        spot=False,
        disk_size=None,
        provider_data={},
    )

    monkeypatch.setattr(provider, "_build_query_variables", lambda: [])
    monkeypatch.setattr(provider, "_fetch_cluster_offers", lambda: [])
    monkeypatch.setattr(provider, "_fetch_cpu_offers", lambda: [cpu_item])

    offers = provider._fetch_offers()

    assert offers == [cpu_item]
