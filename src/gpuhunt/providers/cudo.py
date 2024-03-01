import logging
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from typing import List, Optional

import requests

from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt._internal.constraints import KNOWN_GPUS
from gpuhunt.providers import AbstractProvider

CpuMemoryGpu = namedtuple("CpuMemoryGpu", ["cpu", "memory", "gpu"])
logger = logging.getLogger(__name__)

API_URL = "https://rest.compute.cudo.org/v1"


class CudoProvider(AbstractProvider):
    NAME = "cudo"

    def get(
        self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        offers = self.fetch_all_vm_types()
        return sorted(offers, key=lambda i: i.price)

    def fetch_all_vm_types(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.fetch_vm_type, cmg.cpu, cmg.memory, cmg.gpu)
                for cmg in GPU_MACHINES
            ]
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.info(
                        f"Unable to find VM type with vCPU: {e.vcpu}, Memory: {e.memory_gib} GiB, GPU: {e.gpu}."
                    )
        return list(chain.from_iterable(results))

    def get_raw_catalog_list(self, vm_machine_type_list, vcpu, memory, gpu: int):
        raw_list = []
        for vm in vm_machine_type_list:
            memory = None
            name = gpu_name(vm["gpuModel"])
            if name is not None:
                memory = get_memory(name)
            if gpu and name is None:
                logger.warning("Skip. Unknown GPU name: %s", vm["gpuModel"])
                continue
            raw = RawCatalogItem(
                instance_name=vm["machineType"],
                location=vm["dataCenterId"],
                spot=False,
                price=round(float(vm["totalPriceHr"]["value"]), 5),
                cpu=vcpu,
                memory=memory,
                gpu_count=gpu,
                gpu_name=name,
                gpu_memory=memory,
                disk_size=None,
            )
            raw_list.append(raw)
        return raw_list

    def fetch_vm_type(self, vcpu, memory_gib, gpu):
        try:
            result = self._list_vm_machine_types(vcpu, memory_gib, gpu)
            return self.get_raw_catalog_list(result, vcpu, memory_gib, gpu)
        except requests.HTTPError as e:
            raise VMTypeFetchError(f"Failed to fetch VM type: {e}", vcpu, memory_gib, gpu)

    def _list_vm_machine_types(self, vcpu, memory_gib, gpu):
        resp = requests.request(
            method="GET",
            url=f"{API_URL}/vms/machine-types?vcpu={vcpu}&memory_gib={memory_gib}&gpu={gpu}",
        )
        if resp.ok:
            data = resp.json()
            return data["hostConfigs"]
        resp.raise_for_status()


class VMTypeFetchError(Exception):
    def __init__(self, message, vcpu, memory_gib, gpu):
        super().__init__(message)
        self.vcpu = vcpu
        self.memory_gib = memory_gib
        self.gpu = gpu

    def __str__(self):
        return f"{super().__str__()} - [vCPU: {self.vcpu}, Memory: {self.memory_gib} GiB, GPU: {self.gpu}]"


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


GPU_MAP = {
    "RTX A4000": "A4000",
    "RTX A4500": "A4500",
    "RTX A5000": "A5000",
    "RTX A6000": "A6000",
    "NVIDIA A40": "A40",
    "NVIDIA V100": "V100",
}

GPU_MACHINES = [
    CpuMemoryGpu(1, 1, 1),
    CpuMemoryGpu(1, 2, 1),
    CpuMemoryGpu(1, 3, 1),
    CpuMemoryGpu(1, 4, 1),
    CpuMemoryGpu(2, 2, 1),
    CpuMemoryGpu(2, 2, 2),
    CpuMemoryGpu(2, 3, 1),
    CpuMemoryGpu(2, 3, 2),
    CpuMemoryGpu(2, 4, 1),
    CpuMemoryGpu(2, 4, 2),
    CpuMemoryGpu(2, 6, 1),
    CpuMemoryGpu(2, 6, 2),
    CpuMemoryGpu(2, 8, 1),
    CpuMemoryGpu(2, 8, 2),
    CpuMemoryGpu(3, 3, 1),
    CpuMemoryGpu(3, 3, 2),
    CpuMemoryGpu(3, 3, 3),
    CpuMemoryGpu(3, 4, 1),
    CpuMemoryGpu(3, 4, 2),
    CpuMemoryGpu(3, 4, 3),
    CpuMemoryGpu(3, 6, 1),
    CpuMemoryGpu(3, 6, 2),
    CpuMemoryGpu(3, 6, 3),
    CpuMemoryGpu(3, 8, 1),
    CpuMemoryGpu(3, 8, 2),
    CpuMemoryGpu(3, 8, 3),
    CpuMemoryGpu(3, 12, 1),
    CpuMemoryGpu(3, 12, 2),
    CpuMemoryGpu(3, 12, 3),
    CpuMemoryGpu(4, 4, 1),
    CpuMemoryGpu(4, 4, 2),
    CpuMemoryGpu(4, 4, 3),
    CpuMemoryGpu(4, 4, 4),
    CpuMemoryGpu(4, 6, 1),
    CpuMemoryGpu(4, 6, 2),
    CpuMemoryGpu(4, 6, 3),
    CpuMemoryGpu(4, 6, 4),
    CpuMemoryGpu(4, 8, 1),
    CpuMemoryGpu(4, 8, 2),
    CpuMemoryGpu(4, 8, 3),
    CpuMemoryGpu(4, 8, 4),
    CpuMemoryGpu(4, 12, 1),
    CpuMemoryGpu(4, 12, 2),
    CpuMemoryGpu(4, 12, 3),
    CpuMemoryGpu(4, 12, 4),
    CpuMemoryGpu(4, 16, 1),
    CpuMemoryGpu(4, 16, 2),
    CpuMemoryGpu(4, 16, 3),
    CpuMemoryGpu(4, 16, 4),
    CpuMemoryGpu(6, 6, 1),
    CpuMemoryGpu(6, 6, 2),
    CpuMemoryGpu(6, 6, 3),
    CpuMemoryGpu(6, 6, 4),
    CpuMemoryGpu(6, 6, 5),
    CpuMemoryGpu(6, 6, 6),
    CpuMemoryGpu(6, 8, 1),
    CpuMemoryGpu(6, 8, 2),
    CpuMemoryGpu(6, 8, 3),
    CpuMemoryGpu(6, 8, 4),
    CpuMemoryGpu(6, 8, 5),
    CpuMemoryGpu(6, 8, 6),
    CpuMemoryGpu(6, 12, 1),
    CpuMemoryGpu(6, 12, 2),
    CpuMemoryGpu(6, 12, 3),
    CpuMemoryGpu(6, 12, 4),
    CpuMemoryGpu(6, 12, 5),
    CpuMemoryGpu(6, 12, 6),
    CpuMemoryGpu(6, 16, 1),
    CpuMemoryGpu(6, 16, 2),
    CpuMemoryGpu(6, 16, 3),
    CpuMemoryGpu(6, 16, 4),
    CpuMemoryGpu(6, 16, 5),
    CpuMemoryGpu(6, 16, 6),
    CpuMemoryGpu(6, 24, 1),
    CpuMemoryGpu(6, 24, 2),
    CpuMemoryGpu(6, 24, 3),
    CpuMemoryGpu(6, 24, 4),
    CpuMemoryGpu(6, 24, 5),
    CpuMemoryGpu(6, 24, 6),
    CpuMemoryGpu(8, 8, 1),
    CpuMemoryGpu(8, 8, 2),
    CpuMemoryGpu(8, 8, 3),
    CpuMemoryGpu(8, 8, 4),
    CpuMemoryGpu(8, 8, 5),
    CpuMemoryGpu(8, 8, 6),
    CpuMemoryGpu(8, 8, 7),
    CpuMemoryGpu(8, 8, 8),
    CpuMemoryGpu(8, 12, 1),
    CpuMemoryGpu(8, 12, 2),
    CpuMemoryGpu(8, 12, 3),
    CpuMemoryGpu(8, 12, 4),
    CpuMemoryGpu(8, 12, 5),
    CpuMemoryGpu(8, 12, 6),
    CpuMemoryGpu(8, 12, 7),
    CpuMemoryGpu(8, 12, 8),
    CpuMemoryGpu(8, 16, 1),
    CpuMemoryGpu(8, 16, 2),
    CpuMemoryGpu(8, 16, 3),
    CpuMemoryGpu(8, 16, 4),
    CpuMemoryGpu(8, 16, 5),
    CpuMemoryGpu(8, 16, 6),
    CpuMemoryGpu(8, 16, 7),
    CpuMemoryGpu(8, 16, 8),
    CpuMemoryGpu(8, 24, 1),
    CpuMemoryGpu(8, 24, 2),
    CpuMemoryGpu(8, 24, 3),
    CpuMemoryGpu(8, 24, 4),
    CpuMemoryGpu(8, 24, 5),
    CpuMemoryGpu(8, 24, 6),
    CpuMemoryGpu(8, 24, 7),
    CpuMemoryGpu(8, 24, 8),
    CpuMemoryGpu(8, 32, 1),
    CpuMemoryGpu(8, 32, 2),
    CpuMemoryGpu(8, 32, 3),
    CpuMemoryGpu(8, 32, 4),
    CpuMemoryGpu(8, 32, 5),
    CpuMemoryGpu(8, 32, 6),
    CpuMemoryGpu(8, 32, 7),
    CpuMemoryGpu(8, 32, 8),
    CpuMemoryGpu(12, 12, 1),
    CpuMemoryGpu(12, 12, 2),
    CpuMemoryGpu(12, 12, 3),
    CpuMemoryGpu(12, 12, 4),
    CpuMemoryGpu(12, 12, 5),
    CpuMemoryGpu(12, 12, 6),
    CpuMemoryGpu(12, 12, 7),
    CpuMemoryGpu(12, 12, 8),
    CpuMemoryGpu(12, 16, 1),
    CpuMemoryGpu(12, 16, 2),
    CpuMemoryGpu(12, 16, 3),
    CpuMemoryGpu(12, 16, 4),
    CpuMemoryGpu(12, 16, 5),
    CpuMemoryGpu(12, 16, 6),
    CpuMemoryGpu(12, 16, 7),
    CpuMemoryGpu(12, 16, 8),
    CpuMemoryGpu(12, 24, 1),
    CpuMemoryGpu(12, 24, 2),
    CpuMemoryGpu(12, 24, 3),
    CpuMemoryGpu(12, 24, 4),
    CpuMemoryGpu(12, 24, 5),
    CpuMemoryGpu(12, 24, 6),
    CpuMemoryGpu(12, 24, 7),
    CpuMemoryGpu(12, 24, 8),
    CpuMemoryGpu(12, 32, 1),
    CpuMemoryGpu(12, 32, 2),
    CpuMemoryGpu(12, 32, 3),
    CpuMemoryGpu(12, 32, 4),
    CpuMemoryGpu(12, 32, 5),
    CpuMemoryGpu(12, 32, 6),
    CpuMemoryGpu(12, 32, 7),
    CpuMemoryGpu(12, 32, 8),
    CpuMemoryGpu(12, 48, 1),
    CpuMemoryGpu(12, 48, 2),
    CpuMemoryGpu(12, 48, 3),
    CpuMemoryGpu(12, 48, 4),
    CpuMemoryGpu(12, 48, 5),
    CpuMemoryGpu(12, 48, 6),
    CpuMemoryGpu(12, 48, 7),
    CpuMemoryGpu(12, 48, 8),
    CpuMemoryGpu(16, 16, 1),
    CpuMemoryGpu(16, 16, 2),
    CpuMemoryGpu(16, 16, 3),
    CpuMemoryGpu(16, 16, 4),
    CpuMemoryGpu(16, 16, 5),
    CpuMemoryGpu(16, 16, 6),
    CpuMemoryGpu(16, 16, 7),
    CpuMemoryGpu(16, 16, 8),
    CpuMemoryGpu(16, 24, 1),
    CpuMemoryGpu(16, 24, 2),
    CpuMemoryGpu(16, 24, 3),
    CpuMemoryGpu(16, 24, 4),
    CpuMemoryGpu(16, 24, 5),
    CpuMemoryGpu(16, 24, 6),
    CpuMemoryGpu(16, 24, 7),
    CpuMemoryGpu(16, 24, 8),
    CpuMemoryGpu(16, 32, 1),
    CpuMemoryGpu(16, 32, 2),
    CpuMemoryGpu(16, 32, 3),
    CpuMemoryGpu(16, 32, 4),
    CpuMemoryGpu(16, 32, 5),
    CpuMemoryGpu(16, 32, 6),
    CpuMemoryGpu(16, 32, 7),
    CpuMemoryGpu(16, 32, 8),
    CpuMemoryGpu(16, 48, 1),
    CpuMemoryGpu(16, 48, 2),
    CpuMemoryGpu(16, 48, 3),
    CpuMemoryGpu(16, 48, 4),
    CpuMemoryGpu(16, 48, 5),
    CpuMemoryGpu(16, 48, 6),
    CpuMemoryGpu(16, 48, 7),
    CpuMemoryGpu(16, 48, 8),
    CpuMemoryGpu(16, 64, 1),
    CpuMemoryGpu(16, 64, 2),
    CpuMemoryGpu(16, 64, 3),
    CpuMemoryGpu(16, 64, 4),
    CpuMemoryGpu(16, 64, 5),
    CpuMemoryGpu(16, 64, 6),
    CpuMemoryGpu(16, 64, 7),
    CpuMemoryGpu(16, 64, 8),
    CpuMemoryGpu(24, 24, 1),
    CpuMemoryGpu(24, 24, 2),
    CpuMemoryGpu(24, 24, 3),
    CpuMemoryGpu(24, 24, 4),
    CpuMemoryGpu(24, 24, 5),
    CpuMemoryGpu(24, 24, 6),
    CpuMemoryGpu(24, 24, 7),
    CpuMemoryGpu(24, 24, 8),
    CpuMemoryGpu(24, 32, 1),
    CpuMemoryGpu(24, 32, 2),
    CpuMemoryGpu(24, 32, 3),
    CpuMemoryGpu(24, 32, 4),
    CpuMemoryGpu(24, 32, 5),
    CpuMemoryGpu(24, 32, 6),
    CpuMemoryGpu(24, 32, 7),
    CpuMemoryGpu(24, 32, 8),
    CpuMemoryGpu(24, 48, 1),
    CpuMemoryGpu(24, 48, 2),
    CpuMemoryGpu(24, 48, 3),
    CpuMemoryGpu(24, 48, 4),
    CpuMemoryGpu(24, 48, 5),
    CpuMemoryGpu(24, 48, 6),
    CpuMemoryGpu(24, 48, 7),
    CpuMemoryGpu(24, 48, 8),
    CpuMemoryGpu(24, 64, 1),
    CpuMemoryGpu(24, 64, 2),
    CpuMemoryGpu(24, 64, 3),
    CpuMemoryGpu(24, 64, 4),
    CpuMemoryGpu(24, 64, 5),
    CpuMemoryGpu(24, 64, 6),
    CpuMemoryGpu(24, 64, 7),
    CpuMemoryGpu(24, 64, 8),
    CpuMemoryGpu(24, 96, 1),
    CpuMemoryGpu(24, 96, 2),
    CpuMemoryGpu(24, 96, 3),
    CpuMemoryGpu(24, 96, 4),
    CpuMemoryGpu(24, 96, 5),
    CpuMemoryGpu(24, 96, 6),
    CpuMemoryGpu(24, 96, 7),
    CpuMemoryGpu(24, 96, 8),
    CpuMemoryGpu(32, 32, 1),
    CpuMemoryGpu(32, 32, 2),
    CpuMemoryGpu(32, 32, 3),
    CpuMemoryGpu(32, 32, 4),
    CpuMemoryGpu(32, 32, 5),
    CpuMemoryGpu(32, 32, 6),
    CpuMemoryGpu(32, 32, 7),
    CpuMemoryGpu(32, 32, 8),
    CpuMemoryGpu(32, 48, 1),
    CpuMemoryGpu(32, 48, 2),
    CpuMemoryGpu(32, 48, 3),
    CpuMemoryGpu(32, 48, 4),
    CpuMemoryGpu(32, 48, 5),
    CpuMemoryGpu(32, 48, 6),
    CpuMemoryGpu(32, 48, 7),
    CpuMemoryGpu(32, 48, 8),
    CpuMemoryGpu(32, 64, 1),
    CpuMemoryGpu(32, 64, 2),
    CpuMemoryGpu(32, 64, 3),
    CpuMemoryGpu(32, 64, 4),
    CpuMemoryGpu(32, 64, 5),
    CpuMemoryGpu(32, 64, 6),
    CpuMemoryGpu(32, 64, 7),
    CpuMemoryGpu(32, 64, 8),
    CpuMemoryGpu(32, 96, 1),
    CpuMemoryGpu(32, 96, 2),
    CpuMemoryGpu(32, 96, 3),
    CpuMemoryGpu(32, 96, 4),
    CpuMemoryGpu(32, 96, 5),
    CpuMemoryGpu(32, 96, 6),
    CpuMemoryGpu(32, 96, 7),
    CpuMemoryGpu(32, 96, 8),
    CpuMemoryGpu(32, 128, 2),
    CpuMemoryGpu(32, 128, 3),
    CpuMemoryGpu(32, 128, 4),
    CpuMemoryGpu(32, 128, 5),
    CpuMemoryGpu(32, 128, 6),
    CpuMemoryGpu(32, 128, 7),
    CpuMemoryGpu(32, 128, 8),
    CpuMemoryGpu(48, 48, 2),
    CpuMemoryGpu(48, 48, 3),
    CpuMemoryGpu(48, 48, 4),
    CpuMemoryGpu(48, 48, 5),
    CpuMemoryGpu(48, 48, 6),
    CpuMemoryGpu(48, 48, 7),
    CpuMemoryGpu(48, 48, 8),
    CpuMemoryGpu(48, 64, 2),
    CpuMemoryGpu(48, 64, 3),
    CpuMemoryGpu(48, 64, 4),
    CpuMemoryGpu(48, 64, 5),
    CpuMemoryGpu(48, 64, 6),
    CpuMemoryGpu(48, 64, 7),
    CpuMemoryGpu(48, 64, 8),
    CpuMemoryGpu(48, 96, 2),
    CpuMemoryGpu(48, 96, 3),
    CpuMemoryGpu(48, 96, 4),
    CpuMemoryGpu(48, 96, 5),
    CpuMemoryGpu(48, 96, 6),
    CpuMemoryGpu(48, 96, 7),
    CpuMemoryGpu(48, 96, 8),
    CpuMemoryGpu(48, 128, 2),
    CpuMemoryGpu(48, 128, 3),
    CpuMemoryGpu(48, 128, 4),
    CpuMemoryGpu(48, 128, 5),
    CpuMemoryGpu(48, 128, 6),
    CpuMemoryGpu(48, 128, 7),
    CpuMemoryGpu(48, 128, 8),
    CpuMemoryGpu(48, 192, 2),
    CpuMemoryGpu(48, 192, 3),
    CpuMemoryGpu(48, 192, 4),
    CpuMemoryGpu(48, 192, 5),
    CpuMemoryGpu(48, 192, 6),
    CpuMemoryGpu(48, 192, 7),
    CpuMemoryGpu(48, 192, 8),
    CpuMemoryGpu(64, 64, 2),
    CpuMemoryGpu(64, 64, 3),
    CpuMemoryGpu(64, 64, 4),
    CpuMemoryGpu(64, 64, 5),
    CpuMemoryGpu(64, 64, 6),
    CpuMemoryGpu(64, 64, 7),
    CpuMemoryGpu(64, 64, 8),
    CpuMemoryGpu(64, 96, 2),
    CpuMemoryGpu(64, 96, 3),
    CpuMemoryGpu(64, 96, 4),
    CpuMemoryGpu(64, 96, 5),
    CpuMemoryGpu(64, 96, 6),
    CpuMemoryGpu(64, 96, 7),
    CpuMemoryGpu(64, 96, 8),
    CpuMemoryGpu(64, 128, 2),
    CpuMemoryGpu(64, 128, 3),
    CpuMemoryGpu(64, 128, 4),
    CpuMemoryGpu(64, 128, 5),
    CpuMemoryGpu(64, 128, 6),
    CpuMemoryGpu(64, 128, 7),
    CpuMemoryGpu(64, 128, 8),
    CpuMemoryGpu(64, 192, 2),
    CpuMemoryGpu(64, 192, 4),
    CpuMemoryGpu(64, 192, 5),
    CpuMemoryGpu(64, 192, 6),
    CpuMemoryGpu(64, 192, 7),
    CpuMemoryGpu(64, 192, 8),
    CpuMemoryGpu(64, 256, 4),
    CpuMemoryGpu(64, 256, 5),
    CpuMemoryGpu(64, 256, 6),
    CpuMemoryGpu(64, 256, 7),
    CpuMemoryGpu(64, 256, 8),
    CpuMemoryGpu(96, 96, 4),
    CpuMemoryGpu(96, 96, 6),
    CpuMemoryGpu(96, 96, 7),
    CpuMemoryGpu(96, 96, 8),
    CpuMemoryGpu(96, 128, 4),
    CpuMemoryGpu(96, 128, 6),
    CpuMemoryGpu(96, 128, 7),
    CpuMemoryGpu(96, 128, 8),
    CpuMemoryGpu(96, 192, 6),
    CpuMemoryGpu(96, 192, 7),
    CpuMemoryGpu(96, 192, 8),
    CpuMemoryGpu(96, 256, 6),
    CpuMemoryGpu(96, 256, 7),
    CpuMemoryGpu(96, 256, 8),
    CpuMemoryGpu(96, 384, 6),
    CpuMemoryGpu(96, 384, 7),
    CpuMemoryGpu(96, 384, 8),
    CpuMemoryGpu(128, 128, 8),
    CpuMemoryGpu(128, 192, 8),
    CpuMemoryGpu(128, 256, 8),
    CpuMemoryGpu(128, 384, 8),
]
