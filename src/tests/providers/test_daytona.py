import copy
import logging

import pytest
import requests

import gpuhunt.providers.daytona as daytona_module
from gpuhunt import QueryFilter
from gpuhunt._internal.models import AcceleratorVendor, CPUArchitecture
from gpuhunt.providers.daytona import API_URL, PRICING_URL, DaytonaProvider

PAYLOAD = {
    "gpus": [
        {
            "type": "B300",
            "onDemandPricePerHour": 7.1,
            "spotPricePerHour": 4.08,
        },
        {
            "type": "B200",
            "onDemandPricePerHour": 6.25,
            "spotPricePerHour": 3.59,
        },
        {
            "type": "MI355X",
            "onDemandPricePerHour": 5.99,
            "spotPricePerHour": 3.44,
        },
        {
            "type": "H200",
            "onDemandPricePerHour": 4.54,
            "spotPricePerHour": 2.61,
        },
        {
            "type": "H100",
            "onDemandPricePerHour": 3.95,
            "spotPricePerHour": 2.27,
        },
        {
            "type": "RTX-PRO-6000",
            "onDemandPricePerHour": 3.03,
            "spotPricePerHour": 1.74,
        },
        {
            "type": "RTX-5090",
            "onDemandPricePerHour": 1.29,
            "spotPricePerHour": 0.74,
        },
        {
            "type": "RTX-4090",
            "onDemandPricePerHour": 0.99,
            "spotPricePerHour": 0.57,
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
}

ORGANIZATION_ID = "test-organization"
CONTROL_API_PATHS = {
    "identity": "/api-keys/current",
    "capacity": f"/organizations/{ORGANIZATION_ID}/gpu-capacity",
}
REGIONS_PAYLOAD = [{"id": "us"}, {"id": "eu"}]
CAPACITY_PAYLOAD = {
    "capacity": [
        {"gpuType": "H100", "availableOnDemand": 8, "availableSpot": 8},
        {"gpuType": "RTX-PRO-6000", "availableOnDemand": 2, "availableSpot": 0},
        # Billing rates and physical capacity do not imply create API support.
        {"gpuType": "B200", "availableOnDemand": 8, "availableSpot": 8},
    ],
}


@pytest.fixture
def pricing(requests_mock):
    payload = copy.deepcopy(PAYLOAD)
    requests_mock.get(PRICING_URL, json=lambda request, context: payload)
    return payload


@pytest.fixture(autouse=True)
def shared_regions(requests_mock):
    payload = copy.deepcopy(REGIONS_PAYLOAD)
    requests_mock.get(API_URL + "/shared-regions", json=lambda request, context: payload)
    return payload


@pytest.fixture
def control_api(requests_mock):
    def register(api_url=API_URL):
        payloads = {
            "identity": {"organizationId": ORGANIZATION_ID},
            "capacity": copy.deepcopy(CAPACITY_PAYLOAD),
        }
        for name, path in CONTROL_API_PATHS.items():
            requests_mock.get(
                api_url + path,
                json=lambda request, context, name=name: payloads[name],
            )
        return payloads

    return register


def find_offer(offers, gpu_name="H100", gpu_count=1, spot=False):
    return next(
        offer
        for offer in offers
        if (offer.gpu_name, offer.gpu_count, offer.spot) == (gpu_name, gpu_count, spot)
    )


def no_key_warnings(caplog):
    return [record for record in caplog.records if "DAYTONA_API_KEY" in record.message]


def test_public_offers_cover_supported_counts_and_preserve_create_tokens(pricing, requests_mock):
    offers = DaytonaProvider().get()

    assert len(offers) == 7 * 8 * 2 + 2
    gpu_offers = [offer for offer in offers if offer.gpu_count]
    cpu_offers = [offer for offer in offers if not offer.gpu_count]
    assert {offer.gpu_name for offer in gpu_offers} == {
        "B300",
        "H100",
        "H200",
        "MI355X",
        "RTXPRO6000",
        "RTX5090",
        "RTX4090",
    }
    assert {offer.gpu_count for offer in offers} == set(range(9))
    assert {offer.spot for offer in offers} == {False, True}
    assert offers == sorted(offers, key=lambda offer: offer.price)
    assert all(offer.provider == "daytona" for offer in offers)
    assert all(offer.location == "earth" for offer in gpu_offers)
    assert {offer.location for offer in cpu_offers} == {"us", "eu"}
    assert all(
        (offer.cpu, offer.memory, offer.disk_size, offer.gpu_count, offer.spot)
        == (1, 1, 3, 0, False)
        for offer in cpu_offers
    )
    assert all(
        offer.gpu_name is None
        and offer.gpu_memory is None
        and offer.gpu_vendor is None
        and offer.provider_data == {}
        for offer in cpu_offers
    )
    expected_metadata = {
        "B300": {},
        "H100": {},
        "H200": {},
        "MI355X": {},
        "RTXPRO6000": {"gpu_type": "RTX-PRO-6000"},
        "RTX5090": {"gpu_type": "RTX-5090"},
        "RTX4090": {"gpu_type": "RTX-4090"},
    }
    for offer in gpu_offers:
        assert offer.gpu_name is not None
        assert offer.provider_data == expected_metadata[offer.gpu_name]
    assert [request.url for request in requests_mock.request_history] == [
        PRICING_URL,
        API_URL + "/shared-regions",
    ]
    assert all(r.headers.get("Authorization") is None for r in requests_mock.request_history)


def test_prices_include_selected_cpu_memory_and_disk_at_each_capacity_rate(pricing):
    offers = DaytonaProvider().get()

    # 3.95 + 8 * 0.0504 + 100 * 0.0162 + 256 * 0.000108.
    h100 = find_offer(offers)
    assert h100.price == 6.000848
    assert (h100.cpu, h100.memory, h100.disk_size) == (8, 100, 256)
    assert (h100.gpu_memory, h100.gpu_vendor) == (80, AcceleratorVendor.NVIDIA)
    assert find_offer(offers, spot=True).price == 3.455872

    consumer = find_offer(offers, "RTX4090")
    assert consumer.price == 2.230848
    assert consumer.memory == 50

    eight_gpus = find_offer(offers, gpu_count=8)
    assert (eight_gpus.cpu, eight_gpus.memory, eight_gpus.disk_size) == (64, 800, 2048)
    assert eight_gpus.price == 48.006784

    amd = find_offer(offers, "MI355X")
    assert (amd.gpu_vendor, amd.gpu_memory) == (AcceleratorVendor.AMD, 288)


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("balance_resources", [False, True])
def test_cpu_only_queries_use_public_regions_without_gpu_auth_requests(
    pricing, requests_mock, authenticated, balance_resources
):
    query = QueryFilter(
        provider=["DAYTONA"],
        cpu_arch=CPUArchitecture.X86,
        max_gpu_count=0,
        spot=False,
        min_price=0.066,
        max_price=0.067,
    )
    provider = DaytonaProvider(api_key="test-key" if authenticated else None)
    offers = provider.get(query_filter=query, balance_resources=balance_resources)

    assert {offer.location for offer in offers} == {"us", "eu"}
    assert all(
        (offer.cpu, offer.memory, offer.disk_size, offer.price) == (1, 1, 3, 0.066924)
        for offer in offers
    )
    assert all(offer.gpu_count == 0 and not offer.spot for offer in offers)
    assert [request.url for request in requests_mock.request_history] == [
        PRICING_URL,
        API_URL + "/shared-regions",
    ]
    assert all(r.headers.get("Authorization") is None for r in requests_mock.request_history)
    assert all(r.timeout == 10 for r in requests_mock.request_history)


@pytest.mark.parametrize(
    ("query", "expected", "price"),
    [
        (QueryFilter(min_cpu=2, min_memory=2, min_disk_size=5), (2, 2, 5), 0.13374),
        (QueryFilter(min_memory=2.1), (1, 3, 3), 0.099324),
        (QueryFilter(max_disk_size=1), (1, 1, 1), 0.066708),
        (
            QueryFilter(
                min_cpu=4,
                max_cpu=4,
                min_memory=3.2,
                max_memory=4.9,
                min_disk_size=5,
                max_disk_size=5,
            ),
            (4, 4, 5),
            0.26694,
        ),
    ],
)
def test_cpu_resources_are_sized_and_all_reserved_disk_is_priced(pricing, query, expected, price):
    query.max_gpu_count = 0
    offers = DaytonaProvider().get(query_filter=query)

    assert len(offers) == 2
    for offer in offers:
        assert (offer.cpu, offer.memory, offer.disk_size) == expected
        assert offer.price == price


def test_cpu_offers_do_not_apply_account_resource_or_region_restrictions(pricing, requests_mock):
    query = QueryFilter(max_gpu_count=0, min_cpu=17, min_memory=193, min_disk_size=513)
    offers = DaytonaProvider(api_key="test-key").get(query_filter=query)

    assert {offer.location for offer in offers} == {"us", "eu"}
    assert all((offer.cpu, offer.memory, offer.disk_size) == (17, 193, 513) for offer in offers)
    assert all(offer.price == 4.038804 for offer in offers)
    assert [request.url for request in requests_mock.request_history] == [
        PRICING_URL,
        API_URL + "/shared-regions",
    ]


def test_cpu_regions_and_resource_prices_refresh_each_query(
    pricing, shared_regions, requests_mock
):
    provider = DaytonaProvider(api_key="test-key")
    query = QueryFilter(max_gpu_count=0)
    assert {offer.location for offer in provider.get(query_filter=query)} == {"us", "eu"}
    shared_regions[:] = [{"id": "us"}, {"id": "ap"}]
    pricing["resources"]["onDemand"]["vcpuPerHour"] = 0.0604
    offers = provider.get(query_filter=query)

    assert {offer.location for offer in offers} == {"us", "ap"}
    assert all(offer.price == 0.076924 for offer in offers)
    assert [request.url for request in requests_mock.request_history] == [
        PRICING_URL,
        API_URL + "/shared-regions",
        PRICING_URL,
        API_URL + "/shared-regions",
    ]


@pytest.mark.parametrize(
    "query",
    [
        QueryFilter(gpu_name=["H100"]),
        QueryFilter(gpu_vendor=AcceleratorVendor.NVIDIA),
        QueryFilter(min_gpu_memory=1),
        QueryFilter(min_total_gpu_memory=1),
        QueryFilter(min_compute_capability=(1, 0)),
        QueryFilter(min_gpu_count=1),
        QueryFilter(spot=True),
        QueryFilter(cpu_arch=CPUArchitecture.ARM),
        QueryFilter(provider=["other"]),
        QueryFilter(max_price=0.06),
        QueryFilter(min_price=0.07),
    ],
)
def test_nonmatching_cpu_queries_skip_region_discovery(pricing, requests_mock, query):
    query.max_gpu_count = 0
    assert DaytonaProvider(api_key="test-key").get(query_filter=query) == []
    assert [request.url for request in requests_mock.request_history] == [PRICING_URL]


def test_explicit_zero_min_gpu_count_preserves_shared_cpu_filter_semantics(pricing):
    query = QueryFilter(
        min_gpu_count=0,
        max_gpu_count=0,
        gpu_name=["H100"],
        gpu_vendor=AcceleratorVendor.NVIDIA,
        min_gpu_memory=80,
        min_total_gpu_memory=80,
        min_compute_capability=(9, 0),
    )
    offers = DaytonaProvider().get(query_filter=query)
    assert {(offer.gpu_count, offer.location) for offer in offers} == {(0, "us"), (0, "eu")}


@pytest.mark.parametrize("status_code", [403, 500])
def test_cpu_region_http_errors_propagate(pricing, requests_mock, status_code):
    requests_mock.get(API_URL + "/shared-regions", status_code=status_code)
    with pytest.raises(requests.HTTPError):
        DaytonaProvider().get(query_filter=QueryFilter(max_gpu_count=0))


@pytest.mark.parametrize("payload", [None, {}, ["us"], [{}], [{"id": 7}], [{"id": ""}]])
def test_malformed_cpu_regions_are_rejected(pricing, requests_mock, payload):
    requests_mock.get(API_URL + "/shared-regions", json=payload)
    with pytest.raises(ValueError):
        DaytonaProvider().get(query_filter=QueryFilter(max_gpu_count=0))


def test_empty_shared_regions_return_no_cpu_offers(pricing, shared_regions):
    shared_regions.clear()
    assert DaytonaProvider().get(query_filter=QueryFilter(max_gpu_count=0)) == []


@pytest.mark.parametrize("api_key", [None, "", "  "])
def test_from_env_without_key_warns_only_on_first_query(pricing, monkeypatch, caplog, api_key):
    if api_key is None:
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DAYTONA_API_KEY", api_key)
    monkeypatch.delenv("DAYTONA_API_URL", raising=False)
    with caplog.at_level(logging.WARNING, logger=daytona_module.__name__):
        provider = DaytonaProvider.from_env()
        assert not caplog.records
        provider.get()
        warnings = no_key_warnings(caplog)
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert "availab" in warnings[0].message.lower()
        provider.get()
        assert len(no_key_warnings(caplog)) == 1


def test_no_key_warning_is_per_provider(pricing, caplog):
    with caplog.at_level(logging.WARNING, logger=daytona_module.__name__):
        first = DaytonaProvider()
        second = DaytonaProvider()
        assert not caplog.records
        first.get()
        first.get()
        second.get()
    assert len(no_key_warnings(caplog)) == 2


def test_from_env_uses_optional_api_key_and_control_api_override(
    pricing, control_api, requests_mock, monkeypatch, caplog, shared_regions
):
    api_url = "https://daytona.example/api"
    control_api(api_url)
    requests_mock.get(api_url + "/shared-regions", json=shared_regions)
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")
    monkeypatch.setenv("DAYTONA_API_URL", api_url + "/")
    with caplog.at_level(logging.WARNING, logger=daytona_module.__name__):
        provider = DaytonaProvider.from_env()
        assert not requests_mock.called
        offers = provider.get()
    assert offers
    assert not no_key_warnings(caplog)
    private_requests = [
        r
        for r in requests_mock.request_history
        if r.url not in (PRICING_URL, api_url + "/shared-regions")
    ]
    assert len(private_requests) == 2
    assert all(r.url.startswith(api_url + "/") for r in private_requests)
    assert all(r.headers["Authorization"] == "Bearer test-key" for r in private_requests)
    regions_request = next(
        r for r in requests_mock.request_history if r.url == api_url + "/shared-regions"
    )
    assert regions_request.headers.get("Authorization") is None
    assert all(r.timeout == 10 for r in requests_mock.request_history)


@pytest.mark.parametrize("authenticated", [False, True])
def test_each_query_uses_current_prices(pricing, control_api, requests_mock, authenticated):
    control_api()
    provider = DaytonaProvider(api_key="test-key" if authenticated else None)
    assert find_offer(provider.get()).price == 6.000848
    next(gpu for gpu in pricing["gpus"] if gpu["type"] == "H100")["onDemandPricePerHour"] = 4.95

    assert find_offer(provider.get()).price == 7.000848
    assert sum(request.url == PRICING_URL for request in requests_mock.request_history) == 2


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (QueryFilter(min_cpu=12, min_memory=125.2, min_disk_size=300), (12, 126, 300)),
        (QueryFilter(max_cpu=4, max_memory=40.5, max_disk_size=128), (4, 40, 128)),
        (
            QueryFilter(
                min_cpu=3,
                max_cpu=4,
                min_memory=30.1,
                max_memory=32,
                min_disk_size=50,
                max_disk_size=64,
            ),
            (4, 32, 64),
        ),
    ],
)
def test_query_resources_are_rounded_and_priced_after_sizing(pricing, query, expected):
    query.gpu_name = ["H100"]
    query.min_gpu_count = query.max_gpu_count = 1
    query.spot = False
    offers = DaytonaProvider().get(query_filter=query)

    assert len(offers) == 1
    offer = offers[0]
    assert (offer.cpu, offer.memory, offer.disk_size) == expected
    cpu, memory, disk = expected
    assert offer.price == round(3.95 + cpu * 0.0504 + memory * 0.0162 + disk * 0.000108, 6)


def test_disabling_balancing_uses_per_sandbox_minimums(pricing):
    provider = DaytonaProvider()
    query = QueryFilter(gpu_name=["H100"], min_gpu_count=3, max_gpu_count=3, spot=False)
    offers = provider.get(query_filter=query, balance_resources=False)
    assert len(offers) == 1
    offer = offers[0]
    assert (offer.cpu, offer.memory, offer.disk_size) == (1, 1, 1)
    assert offer.price == 11.916708

    query.min_cpu, query.min_memory, query.min_disk_size = 5, 13.2, 40
    resized = provider.get(query_filter=query, balance_resources=False)[0]
    assert (resized.cpu, resized.memory, resized.disk_size) == (5, 14, 40)
    assert resized.price == 12.33312


@pytest.mark.parametrize(
    "query",
    [
        QueryFilter(min_cpu=5, max_cpu=4),
        QueryFilter(min_memory=40.5, max_memory=40.7),
        QueryFilter(min_disk_size=129, max_disk_size=128),
        QueryFilter(max_cpu=0),
        QueryFilter(max_memory=0),
        QueryFilter(max_disk_size=0),
        QueryFilter(min_gpu_count=9),
        QueryFilter(min_gpu_count=1, max_gpu_count=0),
        QueryFilter(min_gpu_count=4, max_gpu_count=3),
    ],
)
def test_unsatisfiable_resource_and_gpu_ranges_return_no_offers(pricing, query):
    assert DaytonaProvider().get(query_filter=query) == []


def test_gpu_and_price_filters_are_applied_to_final_offers(pricing):
    query = QueryFilter(
        provider=["DAYTONA"],
        gpu_name=["h100"],
        gpu_vendor=AcceleratorVendor.NVIDIA,
        min_gpu_count=2,
        max_gpu_count=4,
        min_gpu_memory=80,
        max_gpu_memory=80,
        min_total_gpu_memory=200,
        max_total_gpu_memory=300,
        min_price=17,
        max_price=19,
        min_compute_capability=(9, 0),
        max_compute_capability=(9, 0),
        spot=False,
    )
    offers = DaytonaProvider().get(query_filter=query)
    assert len(offers) == 1
    assert (offers[0].gpu_name, offers[0].gpu_count, offers[0].spot) == ("H100", 3, False)
    assert offers[0].price == 18.002544


def test_unknown_and_uncreatable_gpu_types_are_not_advertised(pricing, caplog):
    pricing["gpus"].append(
        {"type": "B400", "onDemandPricePerHour": 9.99, "spotPricePerHour": 5.99}
    )
    with caplog.at_level(logging.WARNING, logger=daytona_module.__name__):
        offers = DaytonaProvider().get(query_filter=QueryFilter(min_gpu_count=1))
    assert len(offers) == 112
    assert not {"B200", "B400"} & {offer.gpu_name for offer in offers}
    assert "B400" in caplog.text


def test_authenticated_offers_follow_capacity_by_gpu_and_purchase_option(pricing, control_api):
    payloads = control_api()
    payloads["capacity"]["capacity"][0].update(availableOnDemand=5, availableSpot=3)
    offers = DaytonaProvider(api_key="test-key").get()

    h100 = [offer for offer in offers if offer.gpu_name == "H100"]
    assert {offer.gpu_count for offer in h100 if not offer.spot} == {1, 2, 3, 4, 5}
    assert {offer.gpu_count for offer in h100 if offer.spot} == {1, 2, 3}
    assert not any(offer.gpu_name == "B200" for offer in offers)
    assert not any(offer.gpu_name == "RTXPRO6000" and offer.spot for offer in offers)
    assert all(offer.provider_data == {} for offer in h100)
    assert find_offer(offers, "RTXPRO6000").provider_data == {"gpu_type": "RTX-PRO-6000"}
    assert find_offer(offers).price == 6.000848
    assert find_offer(offers, spot=True).price == 3.455872


def test_account_restrictions_are_not_queried_or_applied_to_capacity_offers(
    pricing, control_api, requests_mock
):
    control_api()
    organization_url = f"{API_URL}/organizations/{ORGANIZATION_ID}"
    query = QueryFilter(
        gpu_name=["H100"],
        min_gpu_count=8,
        max_gpu_count=8,
        min_cpu=128,
        min_memory=1536,
        min_disk_size=4096,
        spot=False,
    )
    offers = DaytonaProvider(api_key="test-key").get(query_filter=query)

    assert len(offers) == 1
    assert (offers[0].gpu_count, offers[0].spot) == (8, False)
    assert (offers[0].cpu, offers[0].memory, offers[0].disk_size) == (128, 1536, 4096)
    assert [request.url for request in requests_mock.request_history] == [
        PRICING_URL,
        API_URL + "/api-keys/current",
        organization_url + "/gpu-capacity",
    ]


def test_capacity_and_prices_refresh_while_identity_is_cached(pricing, control_api, requests_mock):
    payloads = control_api()
    provider = DaytonaProvider(api_key="test-key")
    query = QueryFilter(min_gpu_count=1)
    assert find_offer(provider.get(query_filter=query), gpu_count=8)
    payloads["capacity"]["capacity"][0].update(availableOnDemand=3, availableSpot=0)
    payloads["capacity"]["capacity"][1].update(availableOnDemand=0)
    offers = provider.get(query_filter=query)

    assert {(offer.gpu_name, offer.gpu_count, offer.spot) for offer in offers} == {
        ("H100", 1, False),
        ("H100", 2, False),
        ("H100", 3, False),
    }
    capacity_url = f"{API_URL}/organizations/{ORGANIZATION_ID}/gpu-capacity"
    assert [request.url for request in requests_mock.request_history] == [
        PRICING_URL,
        API_URL + "/api-keys/current",
        capacity_url,
        PRICING_URL,
        capacity_url,
    ]


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize(
    "query",
    [QueryFilter(min_cpu=17), QueryFilter(min_memory=192.1), QueryFilter(min_disk_size=513)],
)
def test_query_cannot_exceed_published_per_gpu_limits(pricing, control_api, query, authenticated):
    control_api()
    query.min_gpu_count = query.max_gpu_count = 1
    provider = DaytonaProvider(api_key="test-key" if authenticated else None)
    assert provider.get(query_filter=query) == []


def test_empty_gpu_capacity_preserves_cpu_estimates_without_unchecked_gpu_fallback(
    pricing, control_api
):
    payloads = control_api()
    payloads["capacity"]["capacity"] = []
    provider = DaytonaProvider(api_key="test-key")
    offers = provider.get()
    assert {(offer.gpu_count, offer.location) for offer in offers} == {(0, "us"), (0, "eu")}
    assert provider.get(query_filter=QueryFilter(min_gpu_count=1)) == []


@pytest.mark.parametrize("path", CONTROL_API_PATHS.values())
def test_authenticated_http_errors_never_fall_back_to_unchecked_offers(
    pricing, control_api, requests_mock, path
):
    control_api()
    requests_mock.get(API_URL + path, status_code=403)
    with pytest.raises(requests.HTTPError):
        DaytonaProvider(api_key="test-key").get()


@pytest.mark.parametrize("path", CONTROL_API_PATHS.values())
def test_malformed_authenticated_payload_is_rejected(pricing, control_api, requests_mock, path):
    control_api()
    requests_mock.get(API_URL + path, json={})
    with pytest.raises(ValueError):
        DaytonaProvider(api_key="test-key").get()


def test_rate_card_http_error_propagates(requests_mock):
    requests_mock.get(PRICING_URL, status_code=500)
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
def test_malformed_rate_card_is_rejected(requests_mock, payload):
    requests_mock.get(PRICING_URL, json=payload)
    with pytest.raises(ValueError):
        DaytonaProvider().get()
