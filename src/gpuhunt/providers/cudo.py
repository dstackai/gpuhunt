import logging
from collections import namedtuple
from itertools import chain
from math import ceil
from typing import List, Optional, TypeVar, Union

import requests
from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt._internal.constraints import (KNOWN_GPUS, get_compute_capability,
                                           is_between)
from gpuhunt.providers import AbstractProvider

CpuMemoryGpu = namedtuple("CpuMemoryGpu", ["cpu", "memory", "gpu"])
logger = logging.getLogger(__name__)

API_URL = "https://rest.compute.cudo.org/v1"
MIN_CPU = 2
MIN_MEMORY = 8
RAM_PER_VRAM = 2
RAM_DIV = 2
CPU_DIV = 2
RAM_PER_CORE = 4
MIN_DISK_SIZE = 100


class CudoProvider(AbstractProvider):
    NAME = "cudo"

    def get(
            self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        offers = self.fetch_offers(query_filter, balance_resources)
        return sorted(offers, key=lambda i: i.price)

    def fetch_offers(self, query_filter: Optional[QueryFilter], balance_resources) -> List[RawCatalogItem]:
        machine_types = self.list_vm_machine_types()
        if query_filter is not None:
            return self.optimize_offers(machine_types, query_filter, balance_resources)
        else:
            offers = []
            for machine_type in machine_types:
                optimized_specs = optimize_offers_with_gpu(QueryFilter(), machine_type, balance_resources=False)
                raw_catalogs = [get_raw_catalog(machine_type, spec) for spec in optimized_specs]
                offers.append(raw_catalogs)
            return list(chain.from_iterable(offers))

    @staticmethod
    def list_vm_machine_types() -> list[dict]:
        resp = requests.request(
            method="GET",
            url=f"{API_URL}/vms/machine-types-2",
        )
        if resp.ok:
            data = resp.json()
            return data["machineTypes"]
        resp.raise_for_status()

    @staticmethod
    def optimize_offers(machine_types: list[dict], q: QueryFilter, balance_resource) -> List[RawCatalogItem]:
        offers = []
        if any(condition is not None for condition in
               [q.min_gpu_count, q.max_gpu_count, q.min_total_gpu_memory, q.max_total_gpu_memory,
                q.min_gpu_memory, q.max_gpu_memory, q.gpu_name]):
            # filter offers with gpus
            gpu_machine_types = [vm for vm in machine_types if vm['maxGpuFree'] != 0]
            for machine_type in gpu_machine_types:
                machine_type["gpu_name"] = gpu_name(machine_type["gpuModel"])
                machine_type["gpu_memory"] = get_memory(machine_type["gpu_name"])
                if not is_between(machine_type["gpu_memory"], q.min_gpu_memory,
                                  q.max_total_gpu_memory):
                    continue
                if q.gpu_name is not None and machine_type["gpu_name"].lower() not in q.gpu_name:
                    continue
                cc = get_compute_capability(machine_type["gpu_name"])
                if not cc or not is_between(cc, q.min_compute_capability, q.max_compute_capability):
                    continue
                optimized_specs = optimize_offers_with_gpu(q, machine_type, balance_resource)
                raw_catalogs = [get_raw_catalog(machine_type, spec) for spec in optimized_specs]
                offers.append(raw_catalogs)
        else:
            cpu_only_machine_types = [vm for vm in machine_types if vm['maxGpuFree'] == 0]
            for machine_type in cpu_only_machine_types:
                optimized_specs = optimize_offers_no_gpu(q, machine_type, balance_resource)
                raw_catalogs = [get_raw_catalog(machine_type, spec) for spec in optimized_specs]
                offers.append(raw_catalogs)

        return list(chain.from_iterable(offers))


class VMTypeFetchError(Exception):
    def __init__(self, message, vcpu, memory_gib, gpu):
        super().__init__(message)
        self.vcpu = vcpu
        self.memory_gib = memory_gib
        self.gpu = gpu

    def __str__(self):
        return f"{super().__str__()} - [vCPU: {self.vcpu}, Memory: {self.memory_gib} GiB, GPU: {self.gpu}]"


def get_raw_catalog(machine_type, spec):
    raw = RawCatalogItem(
        instance_name=machine_type["machineType"],
        location=machine_type["dataCenterId"],
        spot=False,
        price=(round(float(machine_type["vcpuPriceHr"]["value"]), 5) * spec["cpu"]) +
              (round(float(machine_type["memoryGibPriceHr"]["value"]), 5) * spec["memory"]) +
              (round(float(machine_type["gpuPriceHr"]["value"]), 5) * spec.get("gpu", 0)) +
              (round(float(machine_type["minStorageGibPriceHr"]["value"]), 5) * spec["disk_size"]) +
              (round(float(machine_type["ipv4PriceHr"]["value"]), 5)),
        cpu=spec["cpu"],
        memory=spec["memory"],
        gpu_count=spec.get("gpu", 0),
        gpu_name=machine_type.get("gpu_name", ""),
        gpu_memory=machine_type.get("gpu_memory", 0),
        disk_size=spec["disk_size"],
    )
    return raw


def optimize_offers_with_gpu(q: QueryFilter, machine_type, balance_resources) -> List[dict]:
    # Generate ranges for CPU, GPU, and memory based on the specified minimums, maximums, and available resources
    cpu_range = get_cpu_range(q.min_cpu, q.max_cpu, machine_type["maxVcpuFree"])
    gpu_range = get_gpu_range(q.min_gpu_count, q.max_gpu_count, machine_type["maxGpuFree"])
    memory_range = get_memory_range(q.min_memory, q.max_memory, machine_type["maxMemoryGibFree"])
    min_vcpu_per_memory_gib = machine_type.get("minVcpuPerMemoryGib", 0)
    max_vcpu_per_memory_gib = machine_type.get("maxVcpuPerMemoryGib", float('inf'))
    min_vcpu_per_gpu = machine_type.get("minVcpuPerGpu", 0)
    max_vcpu_per_gpu = machine_type.get("maxVcpuPerGpu", float('inf'))
    unbalanced_specs = []
    for cpu in cpu_range:
        for gpu in gpu_range:
            for memory in memory_range:
                # Check CPU/memory constraints
                if not is_between(cpu, memory * min_vcpu_per_memory_gib, memory * max_vcpu_per_memory_gib):
                    continue

                # Check CPU/GPU constraints
                if gpu > 0:
                    if not is_between(cpu, gpu * min_vcpu_per_gpu, gpu * max_vcpu_per_gpu):
                        continue

                # If all constraints are met, append this combination
                unbalanced_specs.append({"cpu": cpu, "memory": memory, "gpu": gpu})

    # If resource balancing is required, filter combinations to meet the balanced memory requirement
    if balance_resources:
        memory_balanced = [spec for spec in unbalanced_specs
                           if spec["memory"] ==
                           get_balanced_memory(spec["gpu"], machine_type["gpu_memory"], q.max_memory)]
        balanced_specs = memory_balanced
        # Add disk
        balanced_specs = [{"cpu": spec["cpu"],
                           "memory": spec["memory"],
                           "gpu": spec["gpu"],
                           "disk_size": get_balanced_disk_size(machine_type["maxStorageGibFree"],
                                                               spec["memory"],
                                                               spec["gpu"] * machine_type["gpu_memory"],
                                                               q.max_disk_size, q.min_disk_size)}
                          for spec in balanced_specs]
        # Return balanced combinations if any; otherwise, return all combinations
        return balanced_specs

    disk_size = q.min_disk_size if q.min_disk_size is not None else MIN_DISK_SIZE
    # Add disk
    unbalanced_specs = [{"cpu": spec["cpu"],
                         "memory": spec["memory"],
                         "gpu": spec["gpu"],
                         "disk_size": disk_size}
                        for spec in unbalanced_specs]
    return unbalanced_specs


def optimize_offers_no_gpu(q: QueryFilter, machine_type, balance_resource) -> List[dict]:
    # Generate ranges for CPU, memory based on the specified minimums, maximums, and available resources
    cpu_range = get_cpu_range(q.min_cpu, q.max_cpu, machine_type["maxVcpuFree"])
    memory_range = get_memory_range(q.min_memory, q.max_memory, machine_type["maxMemoryGibFree"])

    # Cudo Specific Constraints
    min_vcpu_per_memory_gib = machine_type.get("minVcpuPerMemoryGib", 0)
    max_vcpu_per_memory_gib = machine_type.get("maxVcpuPerMemoryGib", float('inf'))

    unbalanced_specs = []
    for cpu in cpu_range:
        for memory in memory_range:
            # Check CPU/memory constraints
            if not is_between(cpu, memory * min_vcpu_per_memory_gib, memory * max_vcpu_per_memory_gib):
                continue
            # If all constraints are met, append this combination
            unbalanced_specs.append({"cpu": cpu, "memory": memory})

    # If resource balancing is required, filter combinations to meet the balanced memory requirement
    if balance_resource:
        cpu_balanced = [spec for spec in unbalanced_specs
                        if spec["cpu"] ==
                        get_balanced_cpu(spec["memory"], q.max_memory)]

        balanced_specs = cpu_balanced
        # Add disk
        disk_size = q.min_disk_size if q.min_disk_size is not None else MIN_DISK_SIZE
        balanced_specs = [{"cpu": spec["cpu"],
                           "memory": spec["memory"],
                           "disk_size": disk_size}
                          for spec in balanced_specs]
        # Return balanced combinations if any; otherwise, return all combinations
        return balanced_specs

    disk_size = q.min_disk_size if q.min_disk_size is not None else MIN_DISK_SIZE
    # Add disk
    unbalanced_specs = [{"cpu": spec["cpu"],
                         "memory": spec["memory"],
                         "gpu": 0,
                         "disk_size": min_none(machine_type["maxStorageGibFree"], disk_size)}
                        for spec in unbalanced_specs]
    return unbalanced_specs


def get_cpu_range(min_cpu, max_cpu, max_cpu_free):
    cpu_range = range(
        min_cpu if min_cpu is not None else MIN_CPU,
        min(max_cpu if max_cpu is not None else max_cpu_free,
            max_cpu_free) + 1
    )
    return cpu_range


def get_gpu_range(min_gpu_count, max_gpu_count, max_gpu_free):
    gpu_range = range(
        min_gpu_count if min_gpu_count is not None else 1,
        min(max_gpu_count if max_gpu_count is not None else max_gpu_free,
            max_gpu_free) + 1
    )
    return gpu_range


def get_memory_range(min_memory, max_memory, max_memory_gib_free):
    memory_range = range(
        int(min_memory) if min_memory is not None else MIN_MEMORY,
        min(int(max_memory) if max_memory is not None else max_memory_gib_free,
            max_memory_gib_free) + 1
    )
    return memory_range


def get_balanced_memory(gpu_count, gpu_memory, max_memory):
    return min_none(
        round_up(
            RAM_PER_VRAM * gpu_memory * gpu_count, RAM_DIV),
        round_down(max_memory, RAM_DIV))


def get_balanced_cpu(memory, max_cpu):
    return min_none(
        round_up(ceil(memory / RAM_PER_CORE), CPU_DIV),
        round_down(max_cpu, CPU_DIV),  # can be None
    )


def get_balanced_disk_size(available_disk, memory, total_gpu_memory, max_disk_size, min_disk_size):
    return max_none(
        min_none(
            available_disk,
            max(memory, total_gpu_memory),
            max_disk_size,
        ), min_disk_size)


def gpu_name(name: str) -> Optional[str]:
    if not name:
        return None
    result = GPU_MAP.get(name)
    if result is None:
        raise Exception("There is no '%s' in GPU_MAP", name)
    return result


def get_memory(gpu_name: str) -> Optional[int]:
    for gpu in KNOWN_GPUS:
        if gpu.name.lower() == gpu_name.lower():
            return gpu.memory
    raise Exception("There is no '%s' in KNOWN_GPUS", gpu_name)


def round_up(value: Optional[Union[int, float]], step: int) -> Optional[int]:
    if value is None:
        return None
    return round_down(value + step - 1, step)


def round_down(value: Optional[Union[int, float]], step: int) -> Optional[int]:
    if value is None:
        return None
    return value // step * step


T = TypeVar("T", bound=Union[int, float])


def min_none(*args: Optional[T]) -> T:
    return min(v for v in args if v is not None)


def max_none(*args: Optional[T]) -> T:
    return max(v for v in args if v is not None)


GPU_MAP = {
    "RTX A4000": "A4000",
    "RTX A4500": "A4500",
    "RTX A5000": "A5000",
    "RTX A6000": "A6000",
    "NVIDIA A40": "A40",
    "NVIDIA V100": "V100",
    "RTX 3080": "RTX3080"
}