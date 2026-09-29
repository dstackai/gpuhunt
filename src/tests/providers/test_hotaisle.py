import pytest
import requests

from gpuhunt.providers.hotaisle import API_URL, FLAG_BARE_METAL, HotAisleProvider

VM_AVAILABLE_URL = f"{API_URL}/teams/test-team/virtual_machines/available/"
BARE_METAL_AVAILABLE_URL = f"{API_URL}/teams/test-team/bare_metal/available/"

AVAILABLE_VM = {
    "OnDemandPrice": 199,
    "Specs": {
        "cpu_cores": 13,
        "ram_capacity": 240 * 1024**3,
        "disk_capacity": 12288 * 1024**3,
        "cpus": {"count": 1, "model": "Xeon Platinum 8470"},
        "gpus": [{"count": 1, "model": "MI300X"}],
    },
}

AVAILABLE_BARE_METAL = {
    "Quantity": 1,
    "OnDemandPrice": 2712,
    "MinimumReservationMinutes": 480,
    "Specs": {
        "cpu_cores": 104,
        "ram_capacity": 2048 * 1024**3,
        "disk_capacity": 123839994396672,
        "cpus": [
            {
                "count": 2,
                "manufacturer": "Intel",
                "model": "Xeon Platinum 8470",
                "cores": 52,
                "frequency": 2000000000,
            }
        ],
        "gpus": [{"count": 8, "manufacturer": "AMD", "model": "MI300X"}],
    },
}


def test_fetch_offers(requests_mock):
    requests_mock.get(VM_AVAILABLE_URL, json=[AVAILABLE_VM])
    requests_mock.get(BARE_METAL_AVAILABLE_URL, text="null")
    offers = HotAisleProvider(api_key="test-key", team_handle="test-team").fetch_offers()
    assert len(offers) == 1
    offer = offers[0]
    assert offer.instance_name == "vm-mi300x-1"
    assert offer.price == 1.99
    assert offer.cpu == 13
    assert offer.memory == 240
    assert offer.gpu_count == 1
    assert offer.gpu_name == "MI300X"
    assert offer.gpu_memory == 192
    assert offer.flags == []
    assert offer.provider_data == {"vm_specs": AVAILABLE_VM["Specs"]}
    assert requests_mock.last_request.headers["Authorization"] == "Token test-key"


def test_fetch_offers_bare_metal(requests_mock):
    requests_mock.get(VM_AVAILABLE_URL, text="null")
    requests_mock.get(BARE_METAL_AVAILABLE_URL, json=[AVAILABLE_BARE_METAL])
    offers = HotAisleProvider(api_key="test-key", team_handle="test-team").fetch_offers()
    assert len(offers) == 1
    offer = offers[0]
    assert offer.instance_name == "bm-mi300x-8"
    assert offer.price == 27.12
    assert offer.cpu == 104
    assert offer.memory == 2048
    assert offer.gpu_count == 8
    assert offer.gpu_name == "MI300X"
    assert offer.gpu_memory == 192
    assert offer.flags == [FLAG_BARE_METAL]
    assert offer.provider_data == {"bare_metal_specs": AVAILABLE_BARE_METAL["Specs"]}


def test_fetch_offers_none_available(requests_mock):
    requests_mock.get(VM_AVAILABLE_URL, text="null")
    requests_mock.get(BARE_METAL_AVAILABLE_URL, text="null")
    offers = HotAisleProvider(api_key="test-key", team_handle="test-team").fetch_offers()
    assert offers == []


@pytest.mark.parametrize(
    ("failing_url", "working_url", "working_item", "expected_instance_name"),
    [
        (BARE_METAL_AVAILABLE_URL, VM_AVAILABLE_URL, AVAILABLE_VM, "vm-mi300x-1"),
        (VM_AVAILABLE_URL, BARE_METAL_AVAILABLE_URL, AVAILABLE_BARE_METAL, "bm-mi300x-8"),
    ],
    ids=["bare-metal-fails", "vm-fails"],
)
def test_fetch_offers_one_kind_fails(
    requests_mock, caplog, failing_url, working_url, working_item, expected_instance_name
):
    requests_mock.get(failing_url, status_code=500)
    requests_mock.get(working_url, json=[working_item])
    offers = HotAisleProvider(api_key="test-key", team_handle="test-team").fetch_offers()
    assert [offer.instance_name for offer in offers] == [expected_instance_name]
    assert [record.levelname for record in caplog.records] == ["ERROR"]


def test_fetch_offers_all_kinds_fail(requests_mock):
    requests_mock.get(VM_AVAILABLE_URL, status_code=500)
    requests_mock.get(BARE_METAL_AVAILABLE_URL, status_code=500)
    with pytest.raises(requests.HTTPError):
        HotAisleProvider(api_key="test-key", team_handle="test-team").fetch_offers()
