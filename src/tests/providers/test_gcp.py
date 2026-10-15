from unittest.mock import Mock

import pytest

from gpuhunt import AcceleratorVendor, CatalogItem
from gpuhunt.providers import gcp


@pytest.fixture
def offers(monkeypatch: pytest.MonkeyPatch) -> list[CatalogItem]:
    # GCP reports one accelerator even for fractional G4 machines.
    machine_types = [
        gcp.compute_v1.MachineType(
            name=f"g4-standard-{cpu}",
            guest_cpus=cpu,
            memory_mb=cpu * 3840,
            accelerators=[
                {
                    "guest_accelerator_type": "nvidia-rtx-pro-6000",
                    "guest_accelerator_count": gpu_count,
                }
            ],
        )
        for cpu, gpu_count in [(6, 1), (12, 1), (24, 1), (48, 1), (96, 2), (192, 4), (384, 8)]
    ]
    machine_types.extend(
        [
            gcp.compute_v1.MachineType(
                name="g2-standard-4",
                guest_cpus=4,
                memory_mb=16384,
                accelerators=[
                    {"guest_accelerator_type": "nvidia-l4", "guest_accelerator_count": 1}
                ],
            ),
            gcp.compute_v1.MachineType(name="n2-standard-2", guest_cpus=2, memory_mb=8192),
        ]
    )
    machine_types_client = Mock()
    machine_types_client.list.return_value = machine_types
    regions_client = Mock()
    regions_client.list.return_value = [
        gcp.compute_v1.Region(zones=["projects/test-project/zones/us-central1-b"])
    ]
    billing_client = Mock()
    billing_client.list_skus.return_value = []
    monkeypatch.setattr(
        gcp.compute_v1, "MachineTypesClient", Mock(return_value=machine_types_client)
    )
    monkeypatch.setattr(gcp.compute_v1, "AcceleratorTypesClient", Mock())
    monkeypatch.setattr(gcp.compute_v1, "RegionsClient", Mock(return_value=regions_client))
    monkeypatch.setattr(gcp.billing_v1, "CloudCatalogClient", Mock(return_value=billing_client))
    monkeypatch.setattr(gcp, "_make_tpu_offers", Mock(return_value=[]))
    monkeypatch.setattr(
        gcp.Prices,
        "get_instance_price",
        lambda self, machine_type, capacity_type: (
            None if capacity_type is gcp.CapacityType.DWS_CALENDAR_MODE else 1.0
        ),
    )
    return gcp.GCPProvider("test-project").get()


class TestGet:
    @pytest.mark.parametrize(
        ("instance_name", "gpu_name", "gpu_count", "gpu_memory"),
        [
            ("g4-standard-6", "RTXPRO6000", 1, 12.0),
            ("g4-standard-12", "RTXPRO6000", 1, 24.0),
            ("g4-standard-24", "RTXPRO6000", 1, 48.0),
            ("g4-standard-48", "RTXPRO6000", 1, 96.0),
            ("g4-standard-96", "RTXPRO6000", 2, 96.0),
            ("g4-standard-192", "RTXPRO6000", 4, 96.0),
            ("g4-standard-384", "RTXPRO6000", 8, 96.0),
            ("g2-standard-4", "L4", 1, 24.0),
            ("n2-standard-2", None, 0, None),
        ],
    )
    def test_machine_gpu_specs(self, offers, instance_name, gpu_name, gpu_count, gpu_memory):
        machine_offers = [offer for offer in offers if offer.instance_name == instance_name]
        assert machine_offers
        for offer in machine_offers:
            assert offer.gpu_name == gpu_name
            assert offer.gpu_count == gpu_count
            assert offer.gpu_memory == gpu_memory
            assert offer.gpu_vendor == (AcceleratorVendor.NVIDIA if gpu_count else None)
