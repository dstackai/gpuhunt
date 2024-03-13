from itertools import chain
from typing import List

import pytest

from src.gpuhunt.providers.cudo import (
    CudoProvider,
    get_balanced_disk_size,
    get_balanced_memory,
    get_memory,
    gpu_name,
    optimize_offers_with_gpu,
)

from gpuhunt import QueryFilter


@pytest.fixture
def machine_types() -> List[dict]:
    return [{
        "dataCenterId": "br-saopaulo-1",
        "machineType": "cascade-lake",
        "cpuModel": "Cascadelake-Server-noTSX",
        "gpuModel": "RTX 3080",
        "gpuModelId": "nvidia-rtx-3080",
        "minVcpuPerMemoryGib": 0.25,
        "maxVcpuPerMemoryGib": 1,
        "minVcpuPerGpu": 1,
        "maxVcpuPerGpu": 13,
        "vcpuPriceHr": {
            "value": "0.002500"
        },
        "memoryGibPriceHr": {
            "value": "0.003800"
        },
        "gpuPriceHr": {
            "value": "0.05"
        },
        "minStorageGibPriceHr": {
            "value": "0.00013"
        },
        "ipv4PriceHr": {
            "value": "0.005500"
        },
        "maxVcpuFree": 76,
        "totalVcpuFree": 377,
        "maxMemoryGibFree": 227,
        "totalMemoryGibFree": 1132,
        "maxGpuFree": 5,
        "totalGpuFree": 24,
        "maxStorageGibFree": 42420,
        "totalStorageGibFree": 42420
    }]


def test_get_offers_with_query_filter():
    cudo = CudoProvider()
    offers = cudo.get(QueryFilter(min_gpu_count=1, max_gpu_count=1), balance_resources=True)
    print(f'{len(offers)} offers found')
    assert len(offers) >= 1, f'No offers found'


def test_get_offers_no_query_filter():
    cudo = CudoProvider()
    offers = cudo.get(balance_resources=True)
    print(f'{len(offers)} offers found')
    assert len(offers) >= 1, f'No offers found'


def test_optimize_offers(machine_types):
    machine_type = machine_types[0]
    machine_type["gpu_memory"] = get_memory(gpu_name(machine_type["gpuModel"]))
    q = QueryFilter(min_cpu=2, min_gpu_count=1, max_gpu_count=1, min_memory=8)
    balance_resource = True
    available_disk = machine_type["maxStorageGibFree"]
    gpu_memory = get_memory(gpu_name(machine_type["gpuModel"]))
    max_memory = q.max_memory
    max_disk_size = q.max_disk_size
    min_disk_size = q.min_disk_size
    vm_configs = optimize_offers_with_gpu(q, machine_type, balance_resources=balance_resource)

    assert len(vm_configs) >= 1

    for config in vm_configs:
        min_cpus_for_memory = machine_type["minVcpuPerMemoryGib"] * config["memory"]
        max_cpus_for_memory = machine_type["maxVcpuPerMemoryGib"] * config["memory"]
        min_cpus_for_gpu = machine_type["minVcpuPerGpu"] * config["gpu"]
        assert config["cpu"] >= min_cpus_for_memory, \
            f"VM config does not meet the minimum CPU:Memory requirement. Required minimum CPUs: " \
            f"{min_cpus_for_memory}, Found: {config['cpu']}"
        assert config["cpu"] <= max_cpus_for_memory, \
            f"VM config exceeds the maximum CPU:Memory allowance. Allowed maximum CPUs: " \
            f"{max_cpus_for_memory}, Found: {config['cpu']}"
        assert config["cpu"] >= min_cpus_for_gpu, \
            f"VM config does not meet the minimum CPU:GPU requirement. " \
            f"Required minimum CPUs: {min_cpus_for_gpu}, Found: {config['cpu']}"
        # Perform the balance resource checks if balance_resource is True
        if balance_resource:
            expected_memory = get_balanced_memory(config['gpu'], gpu_memory, max_memory)
            expected_disk_size = get_balanced_disk_size(available_disk, config['memory'], config["gpu"] * gpu_memory,
                                                        max_disk_size, min_disk_size)

            assert config['memory'] == expected_memory, \
                f"Memory allocation does not match the expected balanced memory. " \
                f"Expected: {expected_memory}, Found: {config['memory']} in config {config}"
            assert config['disk_size'] == expected_disk_size, \
                f"Disk size allocation does not match the expected balanced disk size. " \
                f"Expected: {expected_disk_size}, Found: {config['disk_size']}"
