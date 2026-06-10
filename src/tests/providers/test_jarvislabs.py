import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog
from gpuhunt._internal.models import QueryFilter
from gpuhunt.providers.jarvislabs import (
    JarvisLabsProvider,
    convert_response_to_raw_catalog_items,
)

SERVER_META_RESPONSE = {
    "server_meta": [
        {
            "gpu_type": "L4",
            "region": "india-noida-01",
            "num_free_devices": 3,
            "effective_num_free_devices": 3,
            "spot_num_free_devices": 2,
            "price_per_hour": 0.44,
            "spot_price": 0.29,
            "vram": "24",
            "cpus_per_gpu": 28,
            "ram_per_gpu": 124,
            "workload_type": "container",
            "num_gpus": "8",
        },
        {
            "gpu_type": "L4",
            "region": "india-noida-01",
            "num_free_devices": 3,
            "effective_num_free_devices": 3,
            "spot_num_free_devices": 2,
            "price_per_hour": 0.44,
            "spot_price": 0.29,
            "vram": "24",
            "cpus_per_gpu": 28,
            "ram_per_gpu": 124,
            "workload_type": "vm",
            "num_gpus": "8",
        },
        {
            "gpu_type": "A100-80GB",
            "region": "india-noida-01",
            "num_free_devices": 1,
            "effective_num_free_devices": 1,
            "spot_num_free_devices": 1,
            "price_per_hour": 1.49,
            "spot_price": 0.89,
            "vram": "80",
            "cpus_per_gpu": 28,
            "ram_per_gpu": 112,
            "workload_type": "vm",
            "num_gpus": "4",
        },
        {
            "gpu_type": "RTX-PRO6000",
            "region": "india-chennai-01",
            "num_free_devices": 2,
            "effective_num_free_devices": 2,
            "spot_num_free_devices": 1,
            "price_per_hour": 1.89,
            "spot_price": 1.19,
            "vram": "96",
            "cpus_per_gpu": 28,
            "ram_per_gpu": 160,
            "workload_type": "vm",
            "num_gpus": "8",
        },
        {
            "gpu_type": "RTX PRO 6000",
            "region": "india-noida-01",
            "num_free_devices": 1,
            "effective_num_free_devices": 1,
            "spot_num_free_devices": 0,
            "price_per_hour": 1.89,
            "spot_price": None,
            "vram": "96",
            "cpus_per_gpu": 28,
            "ram_per_gpu": 160,
            "workload_type": "vm",
            "num_gpus": "8",
        },
        {
            "gpu_type": "H100",
            "region": "europe-01",
            "num_free_devices": 25,
            "effective_num_free_devices": 25,
            "spot_num_free_devices": 25,
            "price_per_hour": 2.99,
            "spot_price": None,
            "vram": "80",
            "cpus_per_gpu": 16,
            "ram_per_gpu": 200,
            "workload_type": None,
        },
        {
            "gpu_type": "H100",
            "region": "unknown-region",
            "num_free_devices": 1,
            "effective_num_free_devices": 1,
            "spot_num_free_devices": 1,
            "price_per_hour": 2.99,
            "spot_price": None,
            "vram": "80",
            "cpus_per_gpu": 16,
            "ram_per_gpu": 200,
            "workload_type": "vm",
        },
    ],
    "cpu_meta": {
        "workload_type": "container",
        "combinations": [
            {
                "vcpus": 4,
                "ram_gb": 16,
                "price": 0.0992,
                "available": True,
                "regions": {
                    "india-noida-01": True,
                    "europe-01": False,
                    "unknown-region": True,
                },
            }
        ],
    },
}


def test_convert_response_to_raw_catalog_items():
    offers = convert_response_to_raw_catalog_items(SERVER_META_RESPONSE)
    assert not any(o.spot for o in offers)

    l4_vm = [o for o in offers if o.gpu_name == "L4" and not o.spot]
    assert [o.gpu_count for o in l4_vm] == [1, 2, 3]
    assert [o.price for o in l4_vm] == [0.44, 0.88, 1.32]
    assert [o.instance_name for o in l4_vm] == ["L4-1x", "L4-2x", "L4-3x"]
    assert all(o.provider_data == {} for o in l4_vm)

    a100 = next(o for o in offers if o.instance_name == "A100-1x" and not o.spot)
    assert a100.gpu_name == "A100"
    assert a100.gpu_memory == 80
    assert a100.location == "india-noida-01"
    assert a100.disk_size is None
    assert a100.provider_data == {"gpu_type": "A100-80GB"}

    rtx_pro_6000 = [o for o in offers if o.gpu_name == "RTXPRO6000" and not o.spot]
    assert [o.gpu_count for o in rtx_pro_6000] == [1, 2, 1]
    assert [o.instance_name for o in rtx_pro_6000] == [
        "RTXPRO6000-1x",
        "RTXPRO6000-2x",
        "RTXPRO6000-1x",
    ]
    assert [o.provider_data for o in rtx_pro_6000] == [
        {"gpu_type": "RTX-PRO6000"},
        {"gpu_type": "RTX-PRO6000"},
        {"gpu_type": "RTX PRO 6000"},
    ]
    assert rtx_pro_6000[0].location == "india-chennai-01"
    assert all(o.gpu_memory == 96 for o in rtx_pro_6000)

    h100 = next(o for o in offers if o.gpu_name == "H100")
    assert h100.gpu_count == 1
    assert h100.location == "europe-01"
    assert h100.provider_data == {}
    assert h100.disk_size is None

    cpu = next(o for o in offers if o.gpu_count == 0)
    assert cpu.instance_name == "cpu-4x16"
    assert cpu.location == "india-noida-01"
    assert cpu.cpu == 4
    assert cpu.memory == 16
    assert cpu.provider_data == {}
    assert cpu.disk_size is None

    assert not any(o.location == "unknown-region" for o in offers)


def test_convert_response_warns_and_skips_unsupported_regions(caplog):
    convert_response_to_raw_catalog_items(SERVER_META_RESPONSE)

    assert "Skipping JarvisLabs GPU VM offer in unsupported region unknown-region" in caplog.text
    assert "Skipping JarvisLabs CPU VM offer in unsupported region unknown-region" in caplog.text


def test_convert_response_skips_unmapped_gpu_types_with_spaces(caplog):
    response = {
        "server_meta": [
            {
                "gpu_type": "RTX A6000",
                "region": "india-noida-01",
                "num_free_devices": 1,
                "price_per_hour": 0.79,
                "vram": "48",
                "cpus_per_gpu": 16,
                "ram_per_gpu": 100,
                "workload_type": "vm",
            },
        ],
    }

    assert convert_response_to_raw_catalog_items(response) == []
    assert "Skipping JarvisLabs GPU offer with unmapped gpu_type: RTX A6000" in caplog.text


def test_convert_response_skips_malformed_specs(caplog):
    response = {
        "server_meta": [
            {
                "gpu_type": "L4",
                "region": "india-noida-01",
                "num_free_devices": "bad",
                "price_per_hour": "bad",
                "vram": "24",
                "cpus_per_gpu": 28,
                "ram_per_gpu": 124,
                "workload_type": "vm",
            },
            {
                "gpu_type": "H100",
                "region": "india-noida-01",
                "num_free_devices": 1,
                "price_per_hour": 2.69,
                "vram": "bad",
                "cpus_per_gpu": 16,
                "ram_per_gpu": 200,
                "workload_type": "vm",
            },
        ],
        "cpu_meta": {
            "combinations": [
                {
                    "vcpus": "bad",
                    "ram_gb": 16,
                    "price": 0.0992,
                    "available": True,
                    "regions": {"india-noida-01": True},
                }
            ]
        },
    }

    offers = convert_response_to_raw_catalog_items(response)

    assert offers == []
    assert "Skipping JarvisLabs GPU offer without price: L4" in caplog.text
    assert "Skipping JarvisLabs GPU offer with unknown VRAM: H100" in caplog.text
    assert "Skipping JarvisLabs CPU offer with incomplete specs" in caplog.text


def test_fetch_offers(requests_mock):
    requests_mock.get("https://api.jarvislabs.test/misc/server_meta", json=SERVER_META_RESPONSE)

    provider = JarvisLabsProvider(api_key="test-token", api_url="https://api.jarvislabs.test")
    offers = provider.fetch_offers()

    assert requests_mock.last_request.headers["Authorization"] == "Bearer test-token"
    assert len(offers) == 9
    assert all(o.disk_size is None for o in offers)

    offers = provider.fetch_offers(query_filter=QueryFilter(min_disk_size=250))
    assert len(offers) == 9
    assert all(o.disk_size is None for o in offers)

    offers = provider.fetch_offers(query_filter=QueryFilter(min_disk_size=50))
    assert len(offers) == 9
    assert all(o.disk_size is None for o in offers)


def test_catalog_query(requests_mock, monkeypatch):
    requests_mock.get("https://api.jarvislabs.test/misc/server_meta", json=SERVER_META_RESPONSE)
    monkeypatch.setattr(internal_catalog, "ONLINE_PROVIDERS", ["jarvislabs"])
    monkeypatch.setattr(internal_catalog, "OFFLINE_PROVIDERS", [])

    catalog = Catalog(balance_resources=False, auto_reload=False)
    catalog.add_provider(
        JarvisLabsProvider(api_key="test-token", api_url="https://api.jarvislabs.test")
    )

    assert len(catalog.query(provider=["jarvislabs"], min_gpu_count=2, gpu_name="L4")) == 2
    assert len(catalog.query(provider=["jarvislabs"], gpu_name="A100", min_gpu_memory=80)) == 1
    assert len(catalog.query(provider=["jarvislabs"], gpu_name="RTXPRO6000")) == 3
    assert len(catalog.query(provider=["jarvislabs"], max_gpu_count=0)) == 1
    assert len(catalog.query(provider=["jarvislabs"], min_disk_size=250)) == 9
    assert len(catalog.query(provider=["jarvislabs"], max_disk_size=50)) == 9
    assert len(catalog.query(provider=["jarvislabs"], gpu_name="L4", spot=False)) == 3
    assert len(catalog.query(provider=["jarvislabs"], gpu_name="L4", spot=True)) == 0
