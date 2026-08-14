import pytest
import requests

from gpuhunt import MissingCredsError
from gpuhunt.providers import seeweb as seeweb_module
from gpuhunt.providers.seeweb import SeewebProvider, _normalize_gpu


class FakeResponse:
    def __init__(self, payload, error=None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._payload


@pytest.fixture
def plans_payload() -> dict:
    # Mirrors the live /plans shape: ram is in MB, gpu is a count string, gpu_label has a memory
    # suffix, available_regions carries compatible region codes under "location".
    regions = [{"location": "it-fr2"}, {"location": "it-mi2"}]
    return {
        "status": "ok",
        "plans": [
            {
                "name": "ECS1GPU7",
                "cpu": "8",
                "ram": "32768",
                "disk": "500",
                "gpu": "1",
                "gpu_label": "L40s 48GB",
                "hourly_price": 0.85,
                "available": True,
                "id": 7,
                "host_type": "ECS",
                "available_regions": regions,
            },
            {
                "name": "ECS2GPU3",
                "cpu": "32",
                "ram": "245760",
                "disk": "2000",
                "gpu": "2",
                "gpu_label": "A100 80GB",
                "hourly_price": 1.98,
                "available": True,
                "available_regions": [{"location": "it-fr2"}],
            },
            # CPU-only plan (gpu == "0") must be skipped.
            {
                "name": "eCS4",
                "cpu": "4",
                "ram": "8192",
                "disk": "160",
                "gpu": "0",
                "gpu_label": None,
                "hourly_price": 0.063,
                "available": True,
                "available_regions": regions,
            },
            # Non-NVIDIA accelerator must be skipped by the NVIDIA-only MVP.
            {
                "name": "ECS1GPU9",
                "cpu": "32",
                "ram": "262144",
                "disk": "2000",
                "gpu": "1",
                "gpu_label": "MI300X",
                "hourly_price": 1.6,
                "available": True,
                "available_regions": [{"location": "it-fr2"}],
            },
        ],
    }


def test_get_builds_gpu_offers_per_region(monkeypatch, plans_payload):
    monkeypatch.setattr(
        seeweb_module.requests, "get", lambda *a, **kw: FakeResponse(plans_payload)
    )
    offers = SeewebProvider(token="dummy").get()

    # L40S in 2 regions + A100 in 1 region; CPU-only and MI300X plans skipped.
    assert len(offers) == 3
    assert {o.instance_name for o in offers} == {"ECS1GPU7", "ECS2GPU3"}

    l40s = [o for o in offers if o.instance_name == "ECS1GPU7"]
    assert {o.location for o in l40s} == {"it-fr2", "it-mi2"}
    sample = l40s[0]
    assert sample.gpu_name == "L40S"
    assert sample.gpu_memory == 48
    assert sample.gpu_count == 1
    assert sample.cpu == 8
    assert sample.memory == 32.0  # 32768 MB -> 32 GiB
    assert sample.disk_size == 500.0
    assert sample.gpu_vendor == "nvidia"
    assert sample.spot is False
    assert sample.provider_data == {}

    a100 = next(o for o in offers if o.instance_name == "ECS2GPU3")
    assert a100.gpu_count == 2
    assert a100.gpu_name == "A100"
    assert a100.gpu_memory == 80
    assert a100.memory == 240.0  # 245760 MB -> 240 GiB

    # Offers are sorted by ascending price.
    prices = [o.price for o in offers]
    assert prices == sorted(prices)


def test_get_sends_token_and_timeout(monkeypatch, plans_payload):
    request = {}

    def fake_get(*args, **kwargs):
        request["args"] = args
        request["kwargs"] = kwargs
        return FakeResponse(plans_payload)

    monkeypatch.setattr(seeweb_module.requests, "get", fake_get)
    SeewebProvider(token="dummy").get()

    assert request["args"] == ("https://api.seeweb.it/ecs/v2/plans",)
    assert request["kwargs"]["headers"] == {"X-APITOKEN": "dummy"}
    assert request["kwargs"]["timeout"] == 30


def test_from_env_reads_token(monkeypatch):
    monkeypatch.setenv("SEEWEB_API_TOKEN", "from-env")
    assert SeewebProvider.from_env().token == "from-env"


def test_from_env_without_token_is_rejected(monkeypatch):
    monkeypatch.delenv("SEEWEB_API_TOKEN", raising=False)
    with pytest.raises(MissingCredsError, match="SEEWEB_API_TOKEN"):
        SeewebProvider.from_env()


def test_http_error_is_propagated(monkeypatch):
    error = requests.HTTPError("unauthorized")
    monkeypatch.setattr(
        seeweb_module.requests,
        "get",
        lambda *args, **kwargs: FakeResponse({}, error=error),
    )
    with pytest.raises(requests.HTTPError, match="unauthorized"):
        SeewebProvider(token="dummy").get()


def test_unexpected_response_is_rejected(monkeypatch):
    monkeypatch.setattr(
        seeweb_module.requests,
        "get",
        lambda *args, **kwargs: FakeResponse({"plans": {}}),
    )
    with pytest.raises(ValueError, match="Unexpected response"):
        SeewebProvider(token="dummy").get()


def test_unknown_gpu_label_is_skipped(monkeypatch, caplog):
    payload = {
        "plans": [
            {
                "name": "mystery",
                "cpu": "8",
                "ram": "32768",
                "disk": "100",
                "gpu": "1",
                "gpu_label": "Totally Unknown GPU",
                "hourly_price": 1.0,
                "available_regions": [{"location": "it-fr2"}],
            }
        ]
    }
    monkeypatch.setattr(seeweb_module.requests, "get", lambda *a, **kw: FakeResponse(payload))
    assert SeewebProvider(token="dummy").get() == []
    assert "unknown gpu_label" in caplog.text


def test_inactive_malformed_and_duplicate_regions_are_excluded(monkeypatch, caplog):
    base = {
        "cpu": "8 cores",
        "ram": "32768 MB",
        "disk": "500 GB",
        "gpu": "1 GPU",
        "gpu_label": "NVIDIA L40s 48GB",
        "hourly_price": "$0.85",
        "available": True,
    }
    payload = {
        "plans": [
            {
                **base,
                "name": "active",
                "available_regions": [
                    {"location": "it-fr2"},
                    {"location": "it-fr2"},
                    {"description": "missing code"},
                ],
            },
            {
                **base,
                "name": "inactive",
                "available": False,
                "available_regions": [{"location": "it-fr2"}],
            },
            {
                **base,
                "name": "bad-ram",
                "ram": "unknown",
                "available_regions": [{"location": "it-fr2"}],
            },
            {
                **base,
                "name": "no-regions",
                "available_regions": None,
            },
        ]
    }
    monkeypatch.setattr(seeweb_module.requests, "get", lambda *a, **kw: FakeResponse(payload))

    offers = SeewebProvider(token="dummy").get()

    assert [(offer.instance_name, offer.location) for offer in offers] == [("active", "it-fr2")]
    assert offers[0].cpu == 8
    assert offers[0].memory == 32
    assert offers[0].disk_size == 500
    assert offers[0].price == 0.85
    assert "invalid numeric value" in caplog.text
    assert "invalid available_regions" in caplog.text


def test_normalize_gpu_known_and_mapped():
    # Exact map hits (real Seeweb labels).
    assert _normalize_gpu("A30") == ("A30", 24)
    assert _normalize_gpu("L40s 48GB") == ("L40S", 48)
    assert _normalize_gpu(" NVIDIA   L40S 48GB ") == ("L40S", 48)
    assert _normalize_gpu("RTX A6000 48GB") == ("A6000", 48)
    assert _normalize_gpu("RTX PRO 6000 96GB") == ("RTXPRO6000", 96)
    # find_accelerators fallback with explicit memory parsed from the label.
    assert _normalize_gpu("H100 80GB") == ("H100", 80)
    assert _normalize_gpu("definitely not a gpu") is None
