import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog
from gpuhunt.providers.digitalocean import DigitalOceanProvider

sizes_response = {
    "sizes": [
        {
            "slug": "s-1vcpu-512mb-10gb",
            "memory": 512,
            "vcpus": 1,
            "disk": 10,
            "transfer": 0.5,
            "price_monthly": 4,
            "price_hourly": 0.00595,
            "regions": ["ams3", "fra1", "nyc1", "nyc2", "sfo2", "sfo3", "sgp1", "syd1"],
            "available": True,
            "description": "Basic",
            "networking_throughput": 2000,
            "disk_info": [{"type": "local", "size": {"amount": 10, "unit": "gib"}}],
        },
        {
            "slug": "gpu-h100x8-640gb",
            "memory": 1966080,
            "vcpus": 160,
            "disk": 2046,
            "transfer": 60.0,
            "price_monthly": 17796.48,
            "price_hourly": 23.92,
            "regions": ["nyc1"],
            "available": True,
            "description": "H100 GPU - 8X",
            "networking_throughput": 10000,
            "gpu_info": {
                "count": 8,
                "vram": {"amount": 640, "unit": "gib"},
                "model": "nvidia_h100",
            },
            "disk_info": [
                {"type": "local", "size": {"amount": 2046, "unit": "gib"}},
                {"type": "scratch", "size": {"amount": 40960, "unit": "gib"}},
            ],
        },
        {
            "slug": "gpu-mi300x8-1536gb",
            "memory": 1966080,
            "vcpus": 160,
            "disk": 2046,
            "transfer": 60.0,
            "price_monthly": 11844.48,
            "price_hourly": 15.92,
            "regions": ["nyc1"],
            "available": True,
            "description": "AMD MI300X - 8X",
            "networking_throughput": 10000,
            "gpu_info": {
                "count": 8,
                "vram": {"amount": 1536, "unit": "gib"},
                "model": "amd_mi300x",
            },
            "disk_info": [
                {"type": "local", "size": {"amount": 2046, "unit": "gib"}},
                {"type": "scratch", "size": {"amount": 40960, "unit": "gib"}},
            ],
        },
    ],
    "links": {},
    "meta": {"total": 147},
}


def test_fetch_offers(requests_mock):
    requests_mock.get("https://api.digitalocean.com/v2/sizes", json=sizes_response)

    provider = DigitalOceanProvider(api_key="test-token", api_url="https://api.digitalocean.com")
    offers = provider.fetch_offers()
    assert len(offers) == 10  # 8 CPU offers (8 regions) + 1 NVIDIA + 1 AMD (1 region each)
    catalog = Catalog(balance_resources=False, auto_reload=False)
    digitalocean = DigitalOceanProvider(
        api_key="test-token", api_url="https://api.digitalocean.com"
    )
    internal_catalog.ONLINE_PROVIDERS = ["digitalocean"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(digitalocean)

    # Test queries
    assert (
        len(catalog.query(provider=["digitalocean"], max_gpu_count=0)) == 8
    )  # CPU only (8 regions)
    assert len(catalog.query(provider=["digitalocean"], min_gpu_count=1)) == 2  # GPU instances
    assert len(catalog.query(provider=["digitalocean"], gpu_vendor="nvidia")) == 1  # NVIDIA GPU
    assert len(catalog.query(provider=["digitalocean"], gpu_name="H100")) == 1  # Specific GPU
    assert len(catalog.query(provider=["digitalocean"], gpu_vendor="amd")) == 1  # AMD GPU
    assert len(catalog.query(provider=["digitalocean"], gpu_name="MI300X")) == 1  # AMD GPU
    assert (
        len(catalog.query(provider=["digitalocean"], min_gpu_memory=1000)) == 1
    )  # MI300X: 1536GB
