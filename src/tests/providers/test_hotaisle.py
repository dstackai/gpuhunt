from gpuhunt.providers.hotaisle import API_URL, HotAisleProvider

AVAILABLE_URL = f"{API_URL}/teams/test-team/virtual_machines/available/"

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


def test_fetch_offers(requests_mock):
    requests_mock.get(AVAILABLE_URL, json=[AVAILABLE_VM])
    offers = HotAisleProvider(api_key="test-key", team_handle="test-team").fetch_offers()
    assert len(offers) == 1
    offer = offers[0]
    assert offer.price == 1.99
    assert offer.cpu == 13
    assert offer.memory == 240
    assert offer.gpu_count == 1
    assert offer.gpu_name == "MI300X"
    assert offer.gpu_memory == 192
    assert requests_mock.last_request.headers["Authorization"] == "Token test-key"


def test_fetch_offers_none_available(requests_mock):
    requests_mock.get(AVAILABLE_URL, text="null")
    offers = HotAisleProvider(api_key="test-key", team_handle="test-team").fetch_offers()
    assert offers == []
