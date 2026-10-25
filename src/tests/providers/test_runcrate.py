import copy

import pytest
import requests

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog, MissingCredsError
from gpuhunt._internal.constraints import find_accelerators
from gpuhunt._internal.models import AcceleratorVendor, CPUArchitecture
from gpuhunt.providers.runcrate import GPU_MAP, INSTANCE_TYPES_URL, TIMEOUT, RuncrateProvider

# Recorded from GET /api/v1/instances/types on 2026-10-25 (trimmed to nine of 152 offers, and
# `subscription_terms` to one entry). Offer ids are replaced with placeholders.
RECORDED_RESPONSE = {
    "data": [
        {
            "id": "offer-1-cpu",
            "gpu_type": "CPU",
            "gpu_count": 0,
            "cpu_cores": 4,
            "memory_gb": 16,
            "storage_gb": 250,
            "region": "helsinki-finland-5",
            "deployment_type": "vm",
            "hourly_rate": 0.16,
            "subscription_terms": [
                {
                    "term_months": 1,
                    "discount_pct": 0.05,
                    "effective_hourly_rate": 0.152,
                    "term_hours": 730,
                    "total_price": 110.96,
                    "monthly_equivalent": 110.96,
                    "saving": 5.84,
                }
            ],
        },
        {
            "id": "offer-2-a6000",
            "gpu_type": "A6000",
            "gpu_count": 1,
            "cpu_cores": 6,
            "memory_gb": 48,
            "storage_gb": 256,
            "region": "wichita-usa-1",
            "deployment_type": "vm",
            "hourly_rate": 0.74,
        },
        {
            "id": "offer-3-a6000",
            "gpu_type": "A6000",
            "gpu_count": 4,
            "cpu_cores": 30,
            "memory_gb": 192,
            "storage_gb": 1024,
            "region": "desmoines-usa-1",
            "deployment_type": "vm",
            "hourly_rate": 2.96,
        },
        {
            "id": "offer-4-a100-80g",
            "gpu_type": "A100_80G",
            "gpu_count": 8,
            "cpu_cores": 240,
            "memory_gb": 1280,
            "storage_gb": 9578,
            "region": "omaha-usa-2",
            "deployment_type": "vm",
            "hourly_rate": 13.84,
        },
        {
            "id": "offer-5-a100",
            "gpu_type": "A100",
            "gpu_count": 1,
            "cpu_cores": 14,
            "memory_gb": 117,
            "storage_gb": 1700,
            "region": "calgary-canada-1",
            "deployment_type": "vm",
            "hourly_rate": 1.99,
        },
        {
            "id": "offer-6-gh200",
            "gpu_type": "GH200",
            "gpu_count": 1,
            "cpu_cores": 64,
            "memory_gb": 432,
            "storage_gb": 4096,
            "region": "dulles-usa-3",
            "deployment_type": "vm",
            "hourly_rate": 2.98,
        },
        {
            "id": "offer-7-v100",
            "gpu_type": "V100",
            "gpu_count": 4,
            "cpu_cores": 20,
            "memory_gb": 90,
            "storage_gb": 1000,
            "region": "helsinki-finland-1",
            "deployment_type": "vm",
            "hourly_rate": 2.2,
        },
        {
            "id": "offer-8-rtxpro6000",
            "gpu_type": "RTXPro6000",
            "gpu_count": 1,
            "cpu_cores": 16,
            "memory_gb": 192,
            "storage_gb": 750,
            "region": "beltsville-usa-1",
            "deployment_type": "vm",
            "hourly_rate": 2.85,
        },
        {
            "id": "offer-9-h100",
            "gpu_type": "H100",
            "gpu_count": 8,
            "cpu_cores": 252,
            "memory_gb": 1440,
            "storage_gb": 6600,
            "region": "montreal-canada-2",
            "deployment_type": "vm",
            "hourly_rate": 29.52,
        },
    ],
    "meta": {"total": 9},
}


@pytest.fixture
def response() -> dict:
    return copy.deepcopy(RECORDED_RESPONSE)


def make_instance_type(**overrides) -> dict:
    instance_type = copy.deepcopy(RECORDED_RESPONSE["data"][1])
    instance_type.update(overrides)
    return instance_type


def offers_by_id(offers):
    return {o.instance_name: o for o in offers}


class TestGet:
    def test_one_offer_per_instance_type_sorted_by_price(self, requests_mock, response):
        requests_mock.get(INSTANCE_TYPES_URL, json=response)

        offers = RuncrateProvider(api_key="key").get()

        assert len(offers) == 9
        prices = [o.price for o in offers]
        assert prices == sorted(prices)

    def test_sends_api_key(self, requests_mock, response):
        requests_mock.get(INSTANCE_TYPES_URL, json=response)

        RuncrateProvider(api_key="rc_live_test").get()

        assert requests_mock.last_request.headers["Authorization"] == "Bearer rc_live_test"
        assert requests_mock.last_request.timeout == TIMEOUT

    def test_gpu_offer(self, requests_mock, response):
        requests_mock.get(INSTANCE_TYPES_URL, json=response)

        offer = offers_by_id(RuncrateProvider(api_key="key").get())["offer-3-a6000"]

        assert offer.provider == "runcrate"
        assert offer.location == "desmoines-usa-1"
        assert offer.price == 2.96  # the whole VM, not per GPU
        assert offer.cpu == 30
        assert offer.memory == 192
        assert offer.disk_size == 1024
        assert offer.gpu_vendor == AcceleratorVendor.NVIDIA
        assert offer.gpu_count == 4
        assert offer.gpu_name == "A6000"
        assert offer.gpu_memory == 48
        assert offer.spot is False
        assert offer.cpu_arch == CPUArchitecture.X86
        assert offer.flags == []
        assert offer.provider_data == {"gpu_type": "A6000"}

    def test_cpu_offer(self, requests_mock, response):
        requests_mock.get(INSTANCE_TYPES_URL, json=response)

        offer = offers_by_id(RuncrateProvider(api_key="key").get())["offer-1-cpu"]

        assert offer.gpu_count == 0
        assert offer.gpu_name is None
        assert offer.gpu_memory is None
        assert offer.gpu_vendor is None
        assert offer.cpu == 4
        assert offer.memory == 16
        assert offer.price == 0.16

    @pytest.mark.parametrize(
        ("offer_id", "gpu_name", "gpu_memory"),
        [
            ("offer-4-a100-80g", "A100", 80),
            ("offer-5-a100", "A100", 40),
            ("offer-7-v100", "V100", 16),
            ("offer-8-rtxpro6000", "RTXPRO6000", 96),
            ("offer-9-h100", "H100", 80),
        ],
    )
    def test_gpu_type_normalized(self, requests_mock, response, offer_id, gpu_name, gpu_memory):
        requests_mock.get(INSTANCE_TYPES_URL, json=response)

        offer = offers_by_id(RuncrateProvider(api_key="key").get())[offer_id]

        assert offer.gpu_name == gpu_name
        assert offer.gpu_memory == gpu_memory

    def test_superchip_is_arm(self, requests_mock, response):
        requests_mock.get(INSTANCE_TYPES_URL, json=response)

        offer = offers_by_id(RuncrateProvider(api_key="key").get())["offer-6-gh200"]

        assert offer.gpu_name == "GH200"
        assert offer.cpu_arch == CPUArchitecture.ARM

    def test_skips_unknown_gpu_type(self, requests_mock, caplog):
        requests_mock.get(
            INSTANCE_TYPES_URL,
            json={"data": [make_instance_type(id="new", gpu_type="X1000"), make_instance_type()]},
        )

        offers = RuncrateProvider(api_key="key").get()

        assert [o.gpu_name for o in offers] == ["A6000"]
        assert "unknown gpu_type 'X1000'" in caplog.text

    @pytest.mark.parametrize(
        "overrides",
        [
            {"hourly_rate": None},
            {"hourly_rate": "0.74"},
            {"cpu_cores": True},
            {"region": ""},
            {"gpu_count": 1.5},
            {"id": None},
        ],
        ids=repr,
    )
    def test_skips_instance_type_with_missing_fields(self, requests_mock, caplog, overrides):
        requests_mock.get(INSTANCE_TYPES_URL, json={"data": [make_instance_type(**overrides)]})

        assert RuncrateProvider(api_key="key").get() == []
        assert "Skipping Runcrate instance type" in caplog.text

    @pytest.mark.parametrize(
        "overrides",
        [{"hourly_rate": 0}, {"cpu_cores": 0}, {"memory_gb": 0}, {"storage_gb": 0}],
        ids=repr,
    )
    def test_skips_non_positive_resources_or_price(self, requests_mock, overrides):
        requests_mock.get(INSTANCE_TYPES_URL, json={"data": [make_instance_type(**overrides)]})

        assert RuncrateProvider(api_key="key").get() == []

    def test_skips_malformed_instance_types(self, requests_mock, caplog):
        requests_mock.get(
            INSTANCE_TYPES_URL, json={"data": ["not an instance type", make_instance_type()]}
        )

        assert len(RuncrateProvider(api_key="key").get()) == 1
        assert "malformed" in caplog.text

    def test_unexpected_response_is_rejected(self, requests_mock):
        requests_mock.get(INSTANCE_TYPES_URL, json=[make_instance_type()])

        with pytest.raises(ValueError, match="Unexpected response"):
            RuncrateProvider(api_key="key").get()

    def test_http_error_is_propagated(self, requests_mock):
        requests_mock.get(
            INSTANCE_TYPES_URL,
            status_code=401,
            json={"error": {"code": "unauthorized", "message": "Invalid or revoked API key"}},
        )

        with pytest.raises(requests.HTTPError):
            RuncrateProvider(api_key="key").get()


class TestCatalogQuery:
    def test_query_filters_runcrate_offers(self, requests_mock, monkeypatch, response):
        # Other test modules replace the provider lists without restoring them.
        monkeypatch.setattr(internal_catalog, "ONLINE_PROVIDERS", ["runcrate"])
        requests_mock.get(INSTANCE_TYPES_URL, json=response)
        catalog = Catalog(auto_reload=False)
        catalog.add_provider(RuncrateProvider(api_key="key"))

        offers = catalog.query(provider=["runcrate"], gpu_name=["A100"], min_gpu_memory=80)

        assert [o.instance_name for o in offers] == ["offer-4-a100-80g"]


def test_from_env_reads_api_key(monkeypatch):
    monkeypatch.setenv("RUNCRATE_API_KEY", "rc_live_test")

    assert RuncrateProvider.from_env().api_key == "rc_live_test"


def test_from_env_without_api_key_is_rejected(monkeypatch):
    monkeypatch.delenv("RUNCRATE_API_KEY", raising=False)

    with pytest.raises(MissingCredsError):
        RuncrateProvider.from_env()


@pytest.mark.parametrize(("gpu_type", "gpu"), GPU_MAP.items(), ids=list(GPU_MAP))
def test_gpu_map_uses_known_gpus(gpu_type, gpu):
    gpu_name, gpu_memory = gpu
    known = find_accelerators(names=[gpu_name], vendors=[AcceleratorVendor.NVIDIA])
    assert gpu_memory in {info.memory for info in known}
