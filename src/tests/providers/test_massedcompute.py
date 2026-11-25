from gpuhunt import AcceleratorVendor, CatalogItem
from gpuhunt.providers.massedcompute import INVENTORY_URL, MassedComputeProvider


class TestGet:
    def test_available_products(self, requests_mock):
        # Trimmed GET /gpu-inventory response. Capacity spans sites, while prices and
        # host resources are per VM, even when the VM has multiple GPUs.
        gpu = {
            "name": "gpu_2x_pro_6000_blackwell",
            "description": "2x RTX PRO 6000 Blackwell (96GB)",
            "price_cents_per_hour": 438,
            "specs": {"vcpu_count": 30, "memory_gib": 288, "storage_gb": 1450},
        }
        cpu = {
            "name": "cpu_mini_amd_epyc",
            "description": "Mini AMD EPYC",
            "price_cents_per_hour": 12,
            "specs": {"vcpu_count": 8, "memory_gib": 32, "storage_gb": 400},
        }
        requests_mock.get(
            INVENTORY_URL,
            request_headers={"Authorization": "Bearer test-key"},
            json={
                "gpu_inventory": {
                    "gpu": {
                        "instance_type": gpu,
                        "capacity_available": 8,
                        "regions_with_capacity_available": [
                            {"name": "us-central-1"},
                            {"name": "us-central-2"},
                        ],
                    },
                    "cpu": {"instance_type": cpu, "capacity_available": 14},
                    "sold-out": {"instance_type": gpu, "capacity_available": 0},
                    "unpriced": {
                        "instance_type": {**gpu, "price_cents_per_hour": 0},
                        "capacity_available": 1,
                    },
                }
            },
        )

        offers = MassedComputeProvider("test-key").get()

        assert offers == [
            CatalogItem(
                provider="massedcompute",
                instance_name="cpu_mini_amd_epyc",
                location="any",
                price=0.12,
                cpu=8,
                memory=32,
                disk_size=400,
                gpu_count=0,
                gpu_name=None,
                gpu_memory=None,
                spot=False,
            ),
            CatalogItem(
                provider="massedcompute",
                instance_name="gpu_2x_pro_6000_blackwell",
                location="any",
                price=4.38,
                cpu=30,
                memory=288,
                disk_size=1450,
                gpu_count=2,
                gpu_vendor=AcceleratorVendor.NVIDIA,
                gpu_name="RTXPRO6000",
                gpu_memory=96,
                spot=False,
            ),
        ]

    def test_products_without_reported_gpu_memory(self, requests_mock):
        inventory = {}
        for name, description in [
            ("gpu_1x_h100_nvl", "1x H100 NVL"),
            ("gpu_8x_b300_SXM6", "8x B300 SXM6"),
        ]:
            inventory[name] = {
                "instance_type": {
                    "name": name,
                    "description": description,
                    "price_cents_per_hour": 311,
                    "specs": {"vcpu_count": 18, "memory_gib": 128, "storage_gb": 750},
                },
                "capacity_available": 1,
            }
        requests_mock.get(INVENTORY_URL, json={"gpu_inventory": inventory})

        offers = MassedComputeProvider("test-key").get()

        assert [(o.gpu_name, o.gpu_memory) for o in offers] == [("H100NVL", 94)]
