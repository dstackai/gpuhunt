from typing import List

import pytest

import gpuhunt._internal.catalog as internal_catalog
from gpuhunt import Catalog
from gpuhunt.providers.cudo import (
    CudoProvider,
    get_balanced_disk_size,
    get_balanced_memory,
    get_memory,
    gpu_name,
)


@pytest.fixture
def machine_types() -> List[dict]:
    return [
        {
            "dataCenterId": "br-saopaulo-1",
            "machineType": "cascade-lake",
            "cpuModel": "Cascadelake-Server-noTSX",
            "gpuModel": "RTX 3080",
            "gpuModelId": "nvidia-rtx-3080",
            "minVcpuPerMemoryGib": 0.25,
            "maxVcpuPerMemoryGib": 1,
            "minVcpuPerGpu": 1,
            "maxVcpuPerGpu": 13,
            "vcpuPriceHr": {"value": "0.002500"},
            "memoryGibPriceHr": {"value": "0.003800"},
            "gpuPriceHr": {"value": "0.05"},
            "minStorageGibPriceHr": {"value": "0.00013"},
            "ipv4PriceHr": {"value": "0.005500"},
            "maxVcpuFree": 76,
            "totalVcpuFree": 377,
            "maxMemoryGibFree": 227,
            "totalMemoryGibFree": 1132,
            "maxGpuFree": 5,
            "totalGpuFree": 24,
            "maxStorageGibFree": 42420,
            "totalStorageGibFree": 42420,
        },
        {
            "dataCenterId": "no-luster-1",
            "machineType": "epyc-rome-rtx-a5000",
            "cpuModel": "EPYC-Rome",
            "gpuModel": "RTX A5000",
            "gpuModelId": "nvidia-rtx-a5000",
            "minVcpuPerMemoryGib": 0.259109,
            "maxVcpuPerMemoryGib": 1.036437,
            "minVcpuPerGpu": 1,
            "maxVcpuPerGpu": 16,
            "vcpuPriceHr": {"value": "0.002100"},
            "memoryGibPriceHr": {"value": "0.003400"},
            "gpuPriceHr": {"value": "0.520000"},
            "minStorageGibPriceHr": {"value": "0.000107"},
            "ipv4PriceHr": {"value": "0.003500"},
            "renewableEnergy": False,
            "maxVcpuFree": 116,
            "totalVcpuFree": 208,
            "maxMemoryGibFree": 219,
            "totalMemoryGibFree": 390,
            "maxGpuFree": 4,
            "totalGpuFree": 7,
            "maxStorageGibFree": 1170,
            "totalStorageGibFree": 1170,
        },
    ]


def test_get_offers_with_query_filter(mocker, machine_types):
    catalog = Catalog(balance_resources=False, auto_reload=False)
    cudo = CudoProvider()
    cudo.list_vm_machine_types = mocker.Mock(return_value=machine_types)
    internal_catalog.ONLINE_PROVIDERS = ["cudo"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(cudo)
    query_result = catalog.query(provider=["cudo"], min_gpu_count=1, max_gpu_count=1)
    assert len(query_result) >= 1, "No offers found"


def test_get_offers_for_gpu_name(mocker, machine_types):
    catalog = Catalog(balance_resources=True, auto_reload=False)
    cudo = CudoProvider()
    cudo.list_vm_machine_types = mocker.Mock(return_value=machine_types)
    internal_catalog.ONLINE_PROVIDERS = ["cudo"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(cudo)
    query_result = catalog.query(provider=["cudo"], min_gpu_count=1, gpu_name=["A5000"])
    assert len(query_result) >= 1, "No offers found"


def test_get_offers_for_gpu_memory(mocker, machine_types):
    catalog = Catalog(balance_resources=True, auto_reload=False)
    cudo = CudoProvider()
    cudo.list_vm_machine_types = mocker.Mock(return_value=machine_types)
    internal_catalog.ONLINE_PROVIDERS = ["cudo"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(cudo)
    query_result = catalog.query(provider=["cudo"], min_gpu_count=1, min_gpu_memory=16)
    assert len(query_result) >= 1, "No offers found"


def test_get_offers_for_compute_capability(mocker, machine_types):
    catalog = Catalog(balance_resources=True, auto_reload=False)
    cudo = CudoProvider()
    cudo.list_vm_machine_types = mocker.Mock(return_value=machine_types)
    internal_catalog.ONLINE_PROVIDERS = ["cudo"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(cudo)
    query_result = catalog.query(provider=["cudo"], min_gpu_count=1, min_compute_capability=(8, 6))
    assert len(query_result) >= 1, "No offers found"


def test_get_offers_no_query_filter(mocker, machine_types):
    catalog = Catalog(balance_resources=True, auto_reload=False)
    cudo = CudoProvider()
    cudo.list_vm_machine_types = mocker.Mock(return_value=machine_types)
    internal_catalog.ONLINE_PROVIDERS = ["cudo"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(cudo)
    query_result = catalog.query(provider=["cudo"])
    assert len(query_result) >= 1, "No offers found"


def test_optimize_offers_2(mocker, machine_types):
    catalog = Catalog(balance_resources=True, auto_reload=False)
    cudo = CudoProvider()
    cudo.list_vm_machine_types = mocker.Mock(return_value=machine_types[0:1])
    internal_catalog.ONLINE_PROVIDERS = ["cudo"]
    internal_catalog.OFFLINE_PROVIDERS = []
    catalog.add_provider(cudo)
    query_result = catalog.query(
        provider=["cudo"], min_cpu=2, min_gpu_count=1, max_gpu_count=1, min_memory=8
    )
    machine_type = machine_types[0]
    balance_resource = True
    available_disk = machine_type["maxStorageGibFree"]
    gpu_memory = get_memory(gpu_name(machine_type["gpuModel"]))
    max_memory = None
    max_disk_size = None
    min_disk_size = None

    assert len(query_result) >= 1

    for config in query_result:
        min_cpus_for_memory = machine_type["minVcpuPerMemoryGib"] * config.cpu
        max_cpus_for_memory = machine_type["maxVcpuPerMemoryGib"] * config.memory
        min_cpus_for_gpu = machine_type["minVcpuPerGpu"] * config.gpu_count
        assert config.cpu >= min_cpus_for_memory, (
            f"VM config does not meet the minimum CPU:Memory requirement. Required minimum CPUs: "
            f"{min_cpus_for_memory}, Found: {config.cpu}"
        )
        assert config.cpu <= max_cpus_for_memory, (
            f"VM config exceeds the maximum CPU:Memory allowance. Allowed maximum CPUs: "
            f"{max_cpus_for_memory}, Found: {config.cpu}"
        )
        assert config.cpu >= min_cpus_for_gpu, (
            f"VM config does not meet the minimum CPU:GPU requirement. "
            f"Required minimum CPUs: {min_cpus_for_gpu}, Found: {config.cpu}"
        )
        # Perform the balance resource checks if balance_resource is True
        if balance_resource:
            expected_memory = get_balanced_memory(config.gpu_count, gpu_memory, max_memory)
            expected_disk_size = get_balanced_disk_size(
                available_disk,
                config.memory,
                config.gpu_count * gpu_memory,
                max_disk_size,
                min_disk_size,
            )

            assert config.memory == expected_memory, (
                f"Memory allocation does not match the expected balanced memory. "
                f"Expected: {expected_memory}, Found: {config.memory}"
            )
            assert config.disk_size == expected_disk_size, (
                f"Disk size allocation does not match the expected balanced disk size. "
                f"Expected: {expected_disk_size}, Found: {config.disk_size}"
            )
