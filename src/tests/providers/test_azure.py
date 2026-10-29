from types import SimpleNamespace
from unittest.mock import Mock

from gpuhunt.providers.azure import AzureProvider


def resource_sku(name: str, **capabilities: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        resource_type="virtualMachines",
        locations=["eastus"],
        capabilities=[SimpleNamespace(name=k, value=v) for k, v in capabilities.items()],
    )


class TestGetSpecsAndRegions:
    def test_skips_gpu_instance_types_with_unknown_gpu_memory(self):
        provider = AzureProvider(subscription_id="test", credential=Mock())
        provider.client = Mock()
        provider.client.resource_skus.list.return_value = [
            resource_sku("Standard_D2s_v6", vCPUs="2", MemoryGB="8"),
            resource_sku("Standard_NC4as_T4_v3", vCPUs="4", MemoryGB="28", GPUs="1"),
            resource_sku("Standard_NV12s_v3", vCPUs="12", MemoryGB="112", GPUs="1"),
        ]
        specs, regions = provider.get_specs_and_regions()
        assert sorted(specs) == ["Standard_D2s_v6", "Standard_NC4as_T4_v3"]
        assert specs["Standard_NC4as_T4_v3"].gpu_memory == 16.0
        assert regions == {"eastus"}
