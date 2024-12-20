import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog
from gpuhunt.providers.vultr import VultrProvider, fetch_offers

bare_metal = {
    "plans_metal": [
        {
            "id": "vbm-256c-2048gb-8-mi300x-gpu",
            "physical_cpus": 2,
            "cpu_count": 128,
            "cpu_cores": 128,
            "cpu_threads": 256,
            "cpu_model": "EPYC 9534",
            "cpu_mhz": 2450,
            "ram": 2321924,
            "disk": 3576,
            "disk_count": 8,
            "bandwidth": 10240,
            "monthly_cost": 11773.44,
            "hourly_cost": 17.52,
            "monthly_cost_preemptible": 9891.84,
            "hourly_cost_preemptible": 14.72,
            "type": "NVMe",
            "locations": ["ord"],
        },
        {
            "id": "vbm-112c-2048gb-8-h100-gpu",
            "physical_cpus": 2,
            "cpu_count": 112,
            "cpu_cores": 112,
            "cpu_threads": 224,
            "cpu_model": "Platinum 8480+",
            "cpu_mhz": 2000,
            "ram": 2097152,
            "disk": 960,
            "disk_count": 2,
            "bandwidth": 15360,
            "monthly_cost": 16074.24,
            "hourly_cost": 23.92,
            "monthly_cost_preemptible": 12364.8,
            "hourly_cost_preemptible": 18.4,
            "type": "NVMe",
            "locations": ["sea"],
        },
    ]
}

vm_instances = {
    "plans": [
        {
            "id": "vcg-a100-1c-6g-4vram",
            "vcpu_count": 1,
            "ram": 6144,
            "disk": 70,
            "disk_count": 1,
            "bandwidth": 1024,
            "monthly_cost": 90,
            "hourly_cost": 0.123,
            "type": "vcg",
            "locations": ["ewr"],
            "gpu_vram_gb": 4,
            "gpu_type": "NVIDIA_A100",
        },
        {
            "id": "vcg-a100-12c-120g-80vram",
            "vcpu_count": 12,
            "ram": 122880,
            "disk": 1400,
            "disk_count": 1,
            "bandwidth": 10240,
            "monthly_cost": 1750,
            "hourly_cost": 2.397,
            "type": "vcg",
            "locations": ["ewr"],
            "gpu_vram_gb": 80,
            "gpu_type": "NVIDIA_A100",
        },
        {
            "id": "vcg-a100-6c-60g-40vram",
            "vcpu_count": 12,
            "ram": 61440,
            "disk": 1400,
            "disk_count": 1,
            "bandwidth": 10240,
            "monthly_cost": 800,
            "hourly_cost": 1.397,
            "type": "vcg",
            "locations": ["ewr"],
            "gpu_vram_gb": 40,
            "gpu_type": "NVIDIA_A100",
        },
    ]
}


def test_fetch_offers(requests_mock):
    # Mocking the responses for the API endpoints
    requests_mock.get("https://api.vultr.com/v2/plans-metal?per_page=500", json=bare_metal)
    requests_mock.get("https://api.vultr.com/v2/plans?type=all&per_page=500", json=vm_instances)

    # Fetch offers and verify results
    assert len(fetch_offers()) == 5
    catalog = Catalog(balance_resources=False, auto_reload=False)
    vultr = VultrProvider()
    internal_catalog.ONLINE_PROVIDERS = ["vultr"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(vultr)
    assert len(catalog.query(provider=["vultr"], min_gpu_count=1, max_gpu_count=1)) == 3
    assert len(catalog.query(provider=["vultr"], min_gpu_memory=80, max_gpu_count=1)) == 1
    assert len(catalog.query(provider=["vultr"], gpu_vendor="amd")) == 1
    assert len(catalog.query(provider=["vultr"], gpu_name="MI300X")) == 1
