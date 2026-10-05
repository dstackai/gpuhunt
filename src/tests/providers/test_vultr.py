import pytest

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog
from gpuhunt._internal.models import AcceleratorVendor
from gpuhunt.providers.vultr import (
    VultrProvider,
    fetch_offers,
    get_bare_metal_plans,
    get_instance_plans,
)

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

vdm_gpu_plan = {
    # Public /plans response: a GPU plan whose type differs from its ID prefix,
    # and which has none of the gpu_type/gpu_count/gpu_vram_gb fields.
    "id": "vcg-a40-24c-120g-48vram",
    "vcpu_count": 24,
    "ram": 122880,
    "disk": 1400,
    "disk_type": "DEDICATEDMETAL",
    "monthly_cost": 1250,
    "hourly_cost": 1.712,
    "type": "vdm",
    "locations": ["blr"],
    "deploy_ondemand": True,
    "deploy_preemptible": False,
    "gpu_brand": "NVIDIA",
}

vdm_amd_plan = {
    "id": "vcg-mi325x-252c-2872g-1536vram",
    "vcpu_count": 252,
    "ram": 2940928,
    "disk": 14336,
    "hourly_cost": 36.92,
    "type": "vdm",
    "locations": [],
    "deploy_ondemand": False,
    "deploy_preemptible": True,
    "gpu_brand": "AMD",
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


def test_fetch_offers_skips_empty_locations(requests_mock):
    plan = {**vm_instances["plans"][0], "locations": ["ewr", "", "ord"]}
    requests_mock.get(
        "https://api.vultr.com/v2/plans-metal?per_page=500", json={"plans_metal": []}
    )
    requests_mock.get(
        "https://api.vultr.com/v2/plans?type=all&per_page=500", json={"plans": [plan]}
    )

    assert [offer.location for offer in fetch_offers()] == ["ewr", "ord"]


class TestFetchOffers:
    def test_vdm_without_gpu_metadata(self, requests_mock):
        _mock_plans(requests_mock, plans=[vdm_gpu_plan], plans_metal=[])

        [offer] = fetch_offers()

        assert offer.instance_name == "vcg-a40-24c-120g-48vram"
        assert offer.location == "blr"
        assert offer.cpu == 24
        assert offer.memory == 120
        assert offer.disk_size == 1400
        assert offer.price == 1.712
        assert offer.gpu_name == "A40"
        assert offer.gpu_count == 1
        assert offer.gpu_memory == 48
        assert offer.gpu_vendor == AcceleratorVendor.NVIDIA
        assert not offer.spot

    def test_unknown_vdm_plan(self, requests_mock):
        plan = {**vdm_gpu_plan, "id": "vcg-a40-48c-240g-96vram"}
        _mock_plans(requests_mock, plans=[plan], plans_metal=[])

        assert fetch_offers() == []

    def test_vdm_amd_plan(self, requests_mock):
        # Exercise metadata conversion independently of current availability.
        plan = {
            **vdm_amd_plan,
            "locations": ["ord"],
            "deploy_ondemand": True,
        }
        _mock_plans(requests_mock, plans=[plan], plans_metal=[])

        [offer] = fetch_offers()

        assert offer.gpu_vendor == AcceleratorVendor.AMD
        assert offer.gpu_name == "MI325X"
        assert offer.gpu_count == 8
        assert offer.gpu_memory == 256

    @pytest.mark.parametrize(
        ("plan", "is_bare_metal"),
        [
            (vdm_amd_plan, False),
            (vm_instances["plans"][0], False),
            (bare_metal["plans_metal"][1], True),
        ],
    )
    def test_on_demand_plans(self, requests_mock, plan, is_bare_metal):
        plan = {k: v for k, v in plan.items() if k != "deploy_ondemand"}
        plans = [
            {**plan, "locations": ["ewr"], "deploy_ondemand": True},
            {**plan, "locations": ["ord"], "deploy_ondemand": False, "deploy_preemptible": True},
            {**plan, "locations": ["blr"]},
        ]
        _mock_plans(
            requests_mock,
            plans=[] if is_bare_metal else plans,
            plans_metal=plans if is_bare_metal else [],
        )

        assert [offer.location for offer in fetch_offers()] == ["ewr", "blr"]


class TestGetInstancePlans:
    @pytest.mark.parametrize(
        ("gpu_type", "total_vram", "gpu_count", "gpu_memory"),
        [
            ("NVIDIA_A100", 4, 1, 4),
            ("NVIDIA_A100_SXM", 40, 1, 40),
            ("NVIDIA_A100", 160, 2, 80),
        ],
    )
    def test_vcg_gpu_memory(self, gpu_type, total_vram, gpu_count, gpu_memory):
        plan = {
            **vm_instances["plans"][0],
            "gpu_type": gpu_type,
            "gpu_vram_gb": total_vram,
        }

        offer = get_instance_plans(plan, "ewr")

        assert offer is not None
        assert (offer.gpu_count, offer.gpu_memory) == (gpu_count, gpu_memory)


class TestGetBareMetalPlans:
    def test_mi325x_memory(self):
        # The Vultr plan's memory differs from the generic accelerator catalog.
        plan = {
            "id": "vbm-256c-3072gb-8-mi325x-gpu",
            "cpu_threads": 256,
            "ram": 3145728,
            "disk": 3576,
            "hourly_cost": 36.92,
            "gpu_count": 8,
            "gpu_vram_gb": 2048,
        }

        offer = get_bare_metal_plans(plan, "ord")

        assert offer is not None
        assert offer.gpu_vendor == AcceleratorVendor.AMD
        assert offer.gpu_name == "MI325X"
        assert offer.gpu_count == plan["gpu_count"]
        assert offer.gpu_memory == plan["gpu_vram_gb"] / plan["gpu_count"]


def _mock_plans(requests_mock, *, plans: list[dict], plans_metal: list[dict]) -> None:
    requests_mock.get(
        "https://api.vultr.com/v2/plans?type=all&per_page=500", json={"plans": plans}
    )
    requests_mock.get(
        "https://api.vultr.com/v2/plans-metal?per_page=500", json={"plans_metal": plans_metal}
    )
